#pragma once

#include <filesystem>
#include <memory>
#include <mutex>

#include <sqlite3.h>

#include "inspectflow/detection_record.hpp"

class DetectionStore {
public:
    explicit DetectionStore(
        const std::filesystem::path& database_path
    );

    ~DetectionStore() = default;

    // 数据库连接不能被两个对象重复管理
    DetectionStore(const DetectionStore&) = delete;
    DetectionStore& operator=(const DetectionStore&) = delete;

    // 写入一条记录，返回数据库生成的编号。
    std::int64_t insert(const DetectionRecord& record);

    // 按编号查询；不存在时返回 std::nullopt。
    std::optional<StoredDetection> find_by_id(std::int64_t id);

private:
    void initialize_schema();

    // 保护共享连接，以及插入后读取记录编号的完整操作。
    std::mutex mutex_;

    // 对象销毁时，自动调用 sqlite3_close 关闭连接 ｜ db_：我正在操作哪一个数据库
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_{
        nullptr, &sqlite3_close
    };
};
