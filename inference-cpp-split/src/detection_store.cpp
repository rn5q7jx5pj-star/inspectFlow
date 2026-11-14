#include "inspectflow/detection_store.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cassert>

namespace {

// 语句离开作用域时，自动调用 sqlite3_finalize。
using Statement =
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

// 检查返回值，适用于 prepare、bind 等以 SQLITE_OK 表示成功的操作。
void check_ok(sqlite3* db, int rc) {
    if (rc != SQLITE_OK) {
        throw std::runtime_error(sqlite3_errmsg(db));
    }
}

// 将 SQL 文本转换为语句对象。
Statement prepare(sqlite3* db, const char* sql) {
    sqlite3_stmt* raw_stmt = nullptr;

    const int rc = sqlite3_prepare_v2(
        db,
        sql,
        -1,
        &raw_stmt,
        nullptr
    );

    Statement stmt(raw_stmt, &sqlite3_finalize);
    check_ok(db, rc);

    return stmt;
}

// 将字符串复制到 SQLite 参数中。
void bind_text(
    sqlite3* db,
    sqlite3_stmt* stmt,
    int index,
    const std::string& value
) {
    check_ok(
        db,
        sqlite3_bind_text(
            stmt,
            index,
            value.c_str(),
            -1,
            SQLITE_TRANSIENT
        )
    );
}

// 从当前结果行读取文本，并复制为 C++ 字符串。
std::string read_text(sqlite3_stmt* stmt, int column) {
    const auto* text = sqlite3_column_text(stmt, column);

    if (text == nullptr) {
        throw std::runtime_error("Expected non-null text column");
    }

    return std::string(reinterpret_cast<const char*>(text));
}

}  // namespace

DetectionStore::DetectionStore(
    const std::filesystem::path& database_path
) {
    // 1. 检查路径
    if (database_path.empty()) {
        throw std::invalid_argument(
            "Database path must not be empty"
        );
    }

    // 转成绝对路径，便于确认实际使用的是哪个文件
    const auto absolute_path =
        std::filesystem::absolute(database_path);

    // 2. 确保父目录存在
    // 已存在则保持不变；创建失败时会抛出文件系统异常
    std::filesystem::create_directories(
        absolute_path.parent_path()
    );

    // C++17 中 u8string() 返回 UTF-8 编码的 std::string
    const std::string filename = absolute_path.u8string();

    // 3. 打开数据库：不存在时创建
    sqlite3* raw_db = nullptr;

    const int rc = sqlite3_open(
        filename.c_str(),
        &raw_db
    );

    // 无论打开是否成功，先接管可能返回的连接资源
    db_.reset(raw_db);

    // 4. 检查打开结果
    if (rc != SQLITE_OK) {
        const std::string message =
            db_
                ? sqlite3_errmsg(db_.get())
                : sqlite3_errstr(rc);

        throw std::runtime_error(
            "Failed to open database: " + message
        );
    }

    // 5. 创建表；已经存在则不重复创建
    initialize_schema();

    std::cout
        << "Database ready: "
        << absolute_path
        << '\n';
}

void DetectionStore::initialize_schema() {
    const char* sql = R"sql(
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            request_id TEXT NOT NULL,

            model_name TEXT NOT NULL,
            model_version INTEGER,

            pred_score REAL NOT NULL,

            pred_label INTEGER NOT NULL
                CHECK (pred_label IN (0, 1)),

            inference_ms REAL NOT NULL
                CHECK (inference_ms >= 0)
        );
    )sql";

    const int rc = sqlite3_exec(
        db_.get(),
        sql,
        nullptr,
        nullptr,
        nullptr
    );

    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to initialize database schema: ")
            + sqlite3_errmsg(db_.get())
        );
    }
}

std::int64_t DetectionStore::insert(
    const DetectionRecord& record
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 不把无效数值写进数据库。
    if (!std::isfinite(record.pred_score) ||
        !std::isfinite(record.inference_ms) ||
        record.inference_ms < 0) {
        throw std::invalid_argument("Invalid score or duration");
    }

    const char* sql = R"sql(
        INSERT INTO detections (
            request_id,
            model_name,
            model_version,
            pred_score,
            pred_label,
            inference_ms
        )
        VALUES (?, ?, ?, ?, ?, ?);
    )sql";

    auto stmt = prepare(db_.get(), sql);

    bind_text(db_.get(), stmt.get(), 1, record.request_id);
    bind_text(db_.get(), stmt.get(), 2, record.model_name);

    if (record.model_version.has_value()) {
        check_ok(
            db_.get(),
            sqlite3_bind_int64(
                stmt.get(), 3, *record.model_version
            )
        );
    } else {
        check_ok(
            db_.get(),
            sqlite3_bind_null(stmt.get(), 3)
        );
    }

    check_ok(
        db_.get(),
        sqlite3_bind_double(stmt.get(), 4, record.pred_score)
    );

    check_ok(
        db_.get(),
        sqlite3_bind_int(stmt.get(), 5, record.pred_label ? 1 : 0)
    );

    check_ok(
        db_.get(),
        sqlite3_bind_double(stmt.get(), 6, record.inference_ms)
    );

    const int rc = sqlite3_step(stmt.get());

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(db_.get()));
    }

    return sqlite3_last_insert_rowid(db_.get());
}

std::optional<StoredDetection> DetectionStore::find_by_id(
    std::int64_t id
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (id <= 0) {
        throw std::invalid_argument("Detection ID must be positive");
    }

    const char* sql = R"sql(
        SELECT
            detection_id,
            created_at,
            request_id,
            model_name,
            model_version,
            pred_score,
            pred_label,
            inference_ms
        FROM detections
        WHERE detection_id = ?;
    )sql";

    auto stmt = prepare(db_.get(), sql);

    check_ok(
        db_.get(),
        sqlite3_bind_int64(stmt.get(), 1, id)
    );

    const int rc = sqlite3_step(stmt.get());

    if (rc == SQLITE_DONE) {
        return std::nullopt;
    }

    if (rc != SQLITE_ROW) {
        throw std::runtime_error(sqlite3_errmsg(db_.get()));
    }

    StoredDetection result;

    result.detection_id = sqlite3_column_int64(stmt.get(), 0);
    result.record.created_at = read_text(stmt.get(), 1);
    result.record.request_id = read_text(stmt.get(), 2);
    result.record.model_name = read_text(stmt.get(), 3);

    if (sqlite3_column_type(stmt.get(), 4) != SQLITE_NULL) {
        result.record.model_version =
            sqlite3_column_int64(stmt.get(), 4);
    }

    result.record.pred_score =
        sqlite3_column_double(stmt.get(), 5);

    result.record.pred_label =
        sqlite3_column_int(stmt.get(), 6) != 0;

    result.record.inference_ms =
        sqlite3_column_double(stmt.get(), 7);

    return result;
}

bool DetectionLimiter::try_acquire(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(this->active_ < this->max_active_){
        this->active_++;
        return true;
    }
    return false;
}

void DetectionLimiter::release(){
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ > 0);
    this->active_--;
    return;
}

