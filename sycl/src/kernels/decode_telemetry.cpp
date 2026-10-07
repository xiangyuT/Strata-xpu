// Read-only Sysman counters. No GPU events, synchronization or configuration.
#include <level_zero/zes_api.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <vector>

static int64_t raw_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

int main(int argc, char** argv) try {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 600;
    if (seconds <= 0) throw std::runtime_error("positive duration required");
    if (zesInit(0) != ZE_RESULT_SUCCESS) throw std::runtime_error("zesInit failed");
    uint32_t count = 0;
    zesDriverGet(&count, nullptr);
    std::vector<zes_driver_handle_t> drivers(count);
    zesDriverGet(&count, drivers.data());
    std::vector<zes_device_handle_t> devices;
    for (auto driver : drivers) {
        uint32_t n = 0; zesDeviceGet(driver, &n, nullptr);
        std::vector<zes_device_handle_t> found(n); zesDeviceGet(driver, &n, found.data());
        devices.insert(devices.end(), found.begin(), found.end());
    }
    if (devices.size() != 1) throw std::runtime_error("exactly one visible Sysman device required");
    auto d = devices.front();
    zes_device_properties_t properties{}; properties.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    if (zesDeviceGetProperties(d, &properties) != ZE_RESULT_SUCCESS || properties.core.deviceId != 0xe223)
        throw std::runtime_error("B70 Device ID 0xE223 required");
    uint32_t nmem = 0; zesDeviceEnumMemoryModules(d, &nmem, nullptr);
    std::vector<zes_mem_handle_t> memory(nmem); zesDeviceEnumMemoryModules(d, &nmem, memory.data());
    std::printf("{\"kind\":\"environment\",\"device_id\":\"0xE223\",\"memory_modules\":%u,\"clock\":\"CLOCK_MONOTONIC_RAW\",\"interval_ms\":100}\n", nmem);
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < end) {
        const auto begin = raw_ns();
        zes_pci_stats_t pci{};
        const auto pci_result = zesDevicePciGetStats(d, &pci);
        std::printf("{\"kind\":\"sample\",\"begin_raw_ns\":%lld,\"pci\":{\"result\":%u,\"timestamp\":%llu,\"rx_bytes\":%llu,\"tx_bytes\":%llu,\"generation\":%d,\"width\":%d},\"memory\":[",
            (long long)begin, unsigned(pci_result), (unsigned long long)pci.timestamp,
            (unsigned long long)pci.rxCounter, (unsigned long long)pci.txCounter, pci.speed.gen, pci.speed.width);
        for (size_t i = 0; i < memory.size(); ++i) {
            zes_mem_bandwidth_t bandwidth{};
            const auto result = zesMemoryGetBandwidth(memory[i], &bandwidth);
            std::printf("%s{\"module\":%zu,\"result\":%u,\"timestamp\":%llu,\"read_bytes\":%llu,\"write_bytes\":%llu,\"max_bytes_per_s\":%llu}",
                i ? "," : "", i, unsigned(result), (unsigned long long)bandwidth.timestamp,
                (unsigned long long)bandwidth.readCounter, (unsigned long long)bandwidth.writeCounter,
                (unsigned long long)bandwidth.maxBandwidth);
        }
        std::printf("],\"end_raw_ns\":%lld}\n", (long long)raw_ns());
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "decode telemetry: %s\n", error.what()); return 1; }
