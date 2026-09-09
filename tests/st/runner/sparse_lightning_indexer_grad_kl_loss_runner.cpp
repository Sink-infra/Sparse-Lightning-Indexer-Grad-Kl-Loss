#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include "aclnn_sparse_lightning_indexer_grad_kl_loss.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Config = std::unordered_map<std::string, std::string>;

constexpr const char *INPUT_NAMES[] = {
    "query", "key", "queryIndex", "keyIndex", "weights", "sparseIndices",
    "softmaxMax", "softmaxSum", "queryRope", "keyRope",
    "actual_seq_lengths_query", "actual_seq_lengths_key"};
constexpr const char *OUTPUT_NAMES[] = {"dQueryIndex", "dKeyIndex", "dWeights", "loss"};

[[noreturn]] void Fail(const std::string &message)
{
    throw std::runtime_error(message);
}

void Check(aclError status, const std::string &operation)
{
    if (status != ACL_SUCCESS) {
        Fail(operation + " failed with status " + std::to_string(status));
    }
}

void VerifyLoadedOpApi()
{
    const char *expected = std::getenv("SLI_EXPECTED_OPAPI");
    if (!expected) Fail("SLI_EXPECTED_OPAPI is required for artifact verification");
    std::ifstream maps("/proc/self/maps");
    std::string line;
    bool found = false;
    while (std::getline(maps, line)) {
        if (line.find("libcust_opapi.so") == std::string::npos) continue;
        const auto begin = line.find('/');
        if (begin == std::string::npos || line.substr(begin) != expected) {
            Fail("loaded ACLNN library differs from the verified build: " + line);
        }
        found = true;
    }
    if (!found) Fail("custom ACLNN library is absent from process mappings");
    std::cout << "loaded_opapi=" << expected << " identity=PASS" << std::endl;
}

Config ReadConfig(const std::filesystem::path &path)
{
    std::ifstream stream(path);
    if (!stream) {
        Fail("cannot open case descriptor: " + path.string());
    }
    Config config;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0) {
            Fail("invalid descriptor line in " + path.string() + ": " + line);
        }
        config.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return config;
}

const std::string &Required(const Config &config, const std::string &key)
{
    const auto item = config.find(key);
    if (item == config.end()) {
        Fail("case descriptor is missing " + key);
    }
    return item->second;
}

std::vector<int64_t> ParseShape(const std::string &text)
{
    std::vector<int64_t> shape;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (token.empty()) {
            Fail("empty dimension in shape " + text);
        }
        shape.push_back(std::stoll(token));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return shape;
}

std::size_t TypeSize(const std::string &dtype)
{
    if (dtype == "float16" || dtype == "bfloat16") {
        return 2;
    }
    if (dtype == "float32" || dtype == "int32") {
        return 4;
    }
    if (dtype == "int64") {
        return 8;
    }
    Fail("unsupported dtype " + dtype);
}

aclDataType AclType(const std::string &dtype)
{
    if (dtype == "float16") {
        return ACL_FLOAT16;
    }
    if (dtype == "bfloat16") {
        return ACL_BF16;
    }
    if (dtype == "float32") {
        return ACL_FLOAT;
    }
    if (dtype == "int32") {
        return ACL_INT32;
    }
    if (dtype == "int64") {
        return ACL_INT64;
    }
    Fail("unsupported dtype " + dtype);
}

std::size_t TensorBytes(const std::vector<int64_t> &shape, const std::string &dtype)
{
    std::size_t elements = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) {
            Fail("negative dimension in runner fixture");
        }
        if (dimension == 0) {
            return 0;
        }
        if (elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
            Fail("tensor element count overflow");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    if (elements > std::numeric_limits<std::size_t>::max() / TypeSize(dtype)) {
        Fail("tensor byte count overflow");
    }
    return elements * TypeSize(dtype);
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        strides[index - 1] = strides[index] * std::max<int64_t>(shape[index], 1);
    }
    return strides;
}

std::vector<std::uint8_t> ReadBinary(const std::filesystem::path &path, std::size_t expectedBytes)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        Fail("cannot open tensor data: " + path.string());
    }
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::size_t>(size) != expectedBytes) {
        Fail("tensor byte count mismatch: " + path.string());
    }
    std::vector<std::uint8_t> data(expectedBytes);
    stream.seekg(0);
    if (expectedBytes != 0 && !stream.read(reinterpret_cast<char *>(data.data()), expectedBytes)) {
        Fail("cannot read tensor data: " + path.string());
    }
    return data;
}

class DeviceTensor {
public:
    DeviceTensor() = default;
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
        if (allocation_ != nullptr) {
            aclrtFree(allocation_);
        }
    }

    void Create(const std::vector<int64_t> &shape, const std::string &dtype,
                const std::vector<std::uint8_t> *hostData,
                const std::vector<int64_t> &storageShape = {},
                std::size_t outputOffsetBytes = 0, bool guardOutput = false,
                const std::vector<int64_t> &viewStrides = {}, int64_t inputElementOffset = 0)
    {
        shape_ = shape;
        dtype_ = dtype;
        storageShape_ = storageShape.empty() ? shape : storageShape;
        bytes_ = TensorBytes(storageShape_, dtype);
        const auto strides = viewStrides.empty() ? ContiguousStrides(shape_) : viewStrides;
        if (strides.size() != shape.size() || inputElementOffset < 0 ||
            (guardOutput && (!viewStrides.empty() || inputElementOffset != 0))) {
            Fail("invalid input view metadata");
        }
        if (TensorBytes(shape, dtype) != 0) {
            uint64_t last = static_cast<uint64_t>(inputElementOffset);
            const uint64_t capacity = bytes_ / TypeSize(dtype);
            if (last >= capacity) Fail("input view offset exceeds storage");
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                if (strides[axis] < 0 || (strides[axis] != 0 &&
                    static_cast<uint64_t>(shape[axis] - 1) >
                    (capacity - 1 - last) / static_cast<uint64_t>(strides[axis]))) {
                    Fail("input view exceeds storage");
                }
                last += static_cast<uint64_t>(shape[axis] - 1) * strides[axis];
            }
        }
        if (guardOutput && (hostData != nullptr || outputOffsetBytes % TypeSize(dtype) != 0 ||
                            outputOffsetBytes > 4096)) {
            Fail("invalid guarded output view offset");
        }
        prefixBytes_ = guardOutput ? 64 + outputOffsetBytes : 0;
        suffixBytes_ = guardOutput ? 64 : 0;
        if (bytes_ > std::numeric_limits<std::size_t>::max() - prefixBytes_ - suffixBytes_) {
            Fail("guarded output allocation overflow");
        }
        const std::size_t allocatedBytes = std::max<std::size_t>(bytes_ + prefixBytes_ + suffixBytes_, 1);
        Check(aclrtMalloc(&allocation_, allocatedBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc tensor");
        data_ = static_cast<std::uint8_t *>(allocation_) + prefixBytes_;
        if (guardOutput) {
            Check(aclrtMemset(allocation_, allocatedBytes, 0xA5, allocatedBytes), "aclrtMemset guards");
            ResetGuardedValue();
        }
        if (hostData != nullptr && bytes_ != 0) {
            if (hostData->size() != bytes_) {
                Fail("host tensor byte count mismatch");
            }
            Check(aclrtMemcpy(data_, allocatedBytes, hostData->data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy host-to-device");
        } else if (!guardOutput) {
            Check(aclrtMemset(data_, allocatedBytes, 0xFF, allocatedBytes), "poison output");
        }
        void *storageData = data_;
        int64_t elementOffset = inputElementOffset;
        if (guardOutput) {
            storageData = static_cast<std::uint8_t *>(allocation_) + 64;
            elementOffset = outputOffsetBytes / TypeSize(dtype);
            storageShape_ = {static_cast<int64_t>(bytes_ / TypeSize(dtype)) + elementOffset};
        }
        tensor_ = aclCreateTensor(shape_.data(), shape_.size(), AclType(dtype_), strides.data(), elementOffset,
                                  ACL_FORMAT_ND, storageShape_.data(), storageShape_.size(), storageData);
        if (tensor_ == nullptr) {
            Fail("aclCreateTensor failed");
        }
    }

    std::vector<std::uint8_t> CopyToHost() const
    {
        if (prefixBytes_ != 0) {
            std::vector<std::uint8_t> prefix(prefixBytes_), suffix(suffixBytes_);
            Check(aclrtMemcpy(prefix.data(), prefix.size(), allocation_, prefix.size(),
                              ACL_MEMCPY_DEVICE_TO_HOST), "copy prefix guard");
            Check(aclrtMemcpy(suffix.data(), suffix.size(), static_cast<std::uint8_t *>(data_) + bytes_,
                              suffix.size(), ACL_MEMCPY_DEVICE_TO_HOST), "copy suffix guard");
            if (!std::all_of(prefix.begin(), prefix.end(), [](auto value) { return value == 0xA5; }) ||
                !std::all_of(suffix.begin(), suffix.end(), [](auto value) { return value == 0xA5; })) {
                Fail("output view guard overwritten");
            }
        }
        std::vector<std::uint8_t> result(bytes_);
        if (bytes_ != 0) {
            Check(aclrtMemcpy(result.data(), bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy device-to-host");
        }
        return result;
    }

    void ResetGuardedValue()
    {
        if (bytes_ != 0) {
            // Every output dtype represents the all-one bit pattern as NaN.
            // Unwritten zeros must fail instead of inheriting allocator state.
            Check(aclrtMemset(data_, bytes_, 0xFF, bytes_), "poison output view");
        }
    }

    aclTensor *Get() const { return tensor_; }
    const std::string &Dtype() const { return dtype_; }
    std::size_t Bytes() const { return bytes_; }

private:
    void *allocation_ = nullptr;
    void *data_ = nullptr;
    aclTensor *tensor_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> storageShape_;
    std::string dtype_;
    std::size_t bytes_ = 0;
    std::size_t prefixBytes_ = 0;
    std::size_t suffixBytes_ = 0;
};

class Workspace {
public:
    explicit Workspace(std::size_t bytes) : bytes_(bytes)
    {
        if (bytes_ != 0) {
            Check(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        }
    }
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    ~Workspace()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }
    void *Get() const { return data_; }

private:
    void *data_ = nullptr;
    std::size_t bytes_ = 0;
};

class AclSession {
public:
    AclSession() = default;
    AclSession(const AclSession &) = delete;
    AclSession &operator=(const AclSession &) = delete;

    ~AclSession() { Cleanup(); }

    void Initialize(int32_t deviceId)
    {
        deviceId_ = deviceId;
        try {
            Check(aclInit(nullptr), "aclInit");
            initialized_ = true;
            Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
            deviceSet_ = true;
            Check(aclrtCreateContext(&context_, deviceId_), "aclrtCreateContext");
            Check(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
            Check(aclrtSetOpExecuteTimeOut(120), "aclrtSetOpExecuteTimeOut");
            Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
        } catch (...) {
            Cleanup();
            throw;
        }
    }

    aclrtStream Stream() const { return stream_; }

private:
    void Cleanup() noexcept
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (context_ != nullptr) {
            aclrtDestroyContext(context_);
            context_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (initialized_) {
            aclFinalize();
            initialized_ = false;
        }
    }

    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};

float HalfToFloat(std::uint16_t bits)
{
    const float sign = (bits & 0x8000U) != 0 ? -1.0F : 1.0F;
    const unsigned exponent = (bits >> 10U) & 0x1FU;
    const unsigned fraction = bits & 0x03FFU;
    if (exponent == 0) {
        return fraction == 0 ? sign * 0.0F : sign * std::ldexp(static_cast<float>(fraction), -24);
    }
    if (exponent == 0x1FU) {
        return fraction == 0 ? sign * std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0F + static_cast<float>(fraction) / 1024.0F,
                             static_cast<int>(exponent) - 15);
}

float Decode(const std::uint8_t *data, std::size_t index, const std::string &dtype)
{
    if (dtype == "float32") {
        float value = 0.0F;
        std::memcpy(&value, data + index * 4, sizeof(value));
        return value;
    }
    std::uint16_t bits = 0;
    std::memcpy(&bits, data + index * 2, sizeof(bits));
    if (dtype == "float16") {
        return HalfToFloat(bits);
    }
    if (dtype == "bfloat16") {
        std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
        float value = 0.0F;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }
    Fail("cannot decode output dtype " + dtype);
}

struct ErrorMetrics {
    bool passed = true;
    std::size_t errorCount = 0;
    std::size_t elementCount = 0;
    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    std::size_t firstErrorIndex = std::numeric_limits<std::size_t>::max();
    double firstErrorActual = 0.0;
    double firstErrorGolden = 0.0;
};

ErrorMetrics Compare(const std::vector<std::uint8_t> &actual,
                     const std::vector<std::uint8_t> &expected,
                     const std::string &dtype, double rtol, double atol)
{
    ErrorMetrics metrics;
    if (actual.size() != expected.size() || actual.size() % TypeSize(dtype) != 0) {
        metrics.passed = false;
        metrics.maxAbsolute = metrics.maxRelative = std::numeric_limits<double>::infinity();
        return metrics;
    }
    // Use the explicitly supplied tolerance; the statement defines no threshold.
    metrics.elementCount = actual.size() / TypeSize(dtype);
    for (std::size_t index = 0; index < actual.size() / TypeSize(dtype); ++index) {
        const double observed = Decode(actual.data(), index, dtype);
        const double golden = Decode(expected.data(), index, dtype);
        const double absolute = std::abs(observed - golden);
        const double relative = absolute / std::max(std::abs(golden), 1e-12);
        const bool passed = std::isfinite(observed) && std::isfinite(golden) &&
                            absolute <= atol + rtol * std::abs(golden);
        metrics.maxAbsolute = std::max(metrics.maxAbsolute, absolute);
        metrics.maxRelative = std::max(metrics.maxRelative, relative);
        if (!passed && metrics.passed) {
            metrics.firstErrorIndex = index;
            metrics.firstErrorActual = observed;
            metrics.firstErrorGolden = golden;
        }
        metrics.passed = metrics.passed && passed;
        metrics.errorCount += !passed;
    }
    return metrics;
}

struct RunnerOptions {
    int32_t deviceId = 0;
    double rtol = -1.0;
    double atol = -1.0;
    int benchmarkWarmups = 0;
    int benchmarkRepetitions = 0;
    bool workspaceOnly = false;
    bool captureWorkspaceErrors = false;
    bool continueComparisonErrors = false;
    bool showHelp = false;
    int dumpValues = 0;
    std::string lengthApi = SLI_LENGTH_ARRAY_API ? "array" : "tensor";
    std::string coreDtype;
    std::string weightsDtype;
    std::string dataMode;
    std::vector<std::filesystem::path> fixtureDirs;
};

void PrintUsage(std::ostream &stream)
{
    stream
        << "Usage: sparse_lightning_indexer_grad_kl_loss_runner [OPTIONS] --fixture DIR...\n"
        << "\n"
        << "One process initializes ACL/device/context/stream once and executes every fixture.\n"
        << "\n"
        << "Options:\n"
        << "  --fixture DIR          Raw fixture directory; repeatable\n"
        << "  --workspace-only       Query workspace without launching the Kernel\n"
        << "  --length-api MODE      array, tensor, tensor_host, or tensor_as_array (negative probe)\n"
        << "  --capture-workspace-errors  Record preparation errors for every fixture; exit nonzero\n"
        << "  --continue-comparison-errors  Check remaining fixtures after numerical mismatches\n"
        << "  --core-dtype TYPE      Require float16 or bfloat16 fixtures\n"
        << "  --weights-dtype TYPE   Require float16, bfloat16, or float32 weights\n"
        << "  --data-mode MODE       Require empty, zero, or nonzero fixture data\n"
        << "  --benchmark-warmup N   Warm-up launches before device timing\n"
        << "  --benchmark-repeat N   Timed launches; report device-event statistics\n"
        << "  --device ID            NPU device ID (default: 0)\n"
        << "  --rtol VALUE           Explicit relative tolerance for correctness\n"
        << "  --atol VALUE           Explicit absolute tolerance for correctness\n"
        << "  -h, --help             Show this help and exit\n";
}

RunnerOptions ParseArguments(int argc, char **argv)
{
    RunnerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            options.showHelp = true;
        } else if (argument == "--device" && index + 1 < argc) {
            options.deviceId = std::stoi(argv[++index]);
        } else if (argument == "--rtol" && index + 1 < argc) {
            options.rtol = std::stod(argv[++index]);
        } else if (argument == "--atol" && index + 1 < argc) {
            options.atol = std::stod(argv[++index]);
        } else if (argument == "--fixture" && index + 1 < argc) {
            options.fixtureDirs.emplace_back(argv[++index]);
        } else if (argument == "--workspace-only") {
            options.workspaceOnly = true;
        } else if (argument == "--length-api" && index + 1 < argc) {
            options.lengthApi = argv[++index];
        } else if (argument == "--capture-workspace-errors") {
            options.captureWorkspaceErrors = true;
        } else if (argument == "--continue-comparison-errors") {
            options.continueComparisonErrors = true;
        } else if (argument == "--core-dtype" && index + 1 < argc) {
            options.coreDtype = argv[++index];
        } else if (argument == "--weights-dtype" && index + 1 < argc) {
            options.weightsDtype = argv[++index];
        } else if (argument == "--dump-values" && index + 1 < argc) {
            options.dumpValues = std::stoi(argv[++index]);
        } else if (argument == "--data-mode" && index + 1 < argc) {
            options.dataMode = argv[++index];
        } else if (argument == "--benchmark-warmup" && index + 1 < argc) {
            options.benchmarkWarmups = std::stoi(argv[++index]);
        } else if (argument == "--benchmark-repeat" && index + 1 < argc) {
            options.benchmarkRepetitions = std::stoi(argv[++index]);
        } else if (!argument.empty() && argument.front() != '-') {
            options.fixtureDirs.emplace_back(argument);
        } else {
            Fail("unknown or incomplete runner argument: " + argument);
        }
    }
    if (options.showHelp) {
        return options;
    }
    if (options.fixtureDirs.empty()) {
        Fail("runner requires at least one fixture directory");
    }
    if (options.lengthApi != "array" && options.lengthApi != "tensor" && options.lengthApi != "tensor_host" &&
        options.lengthApi != "tensor_as_array" && options.lengthApi != "canonical_tensor") {
        Fail("unknown actual-length API mode");
    }
    if ((options.lengthApi == "tensor_as_array" ||
         (SLI_LENGTH_ARRAY_API && options.lengthApi == "canonical_tensor")) && !options.workspaceOnly) {
        Fail("the negative ABI probe is restricted to Workspace queries");
    }
#if !SLI_LENGTH_ARRAY_API
    if (options.lengthApi == "array" || options.lengthApi == "tensor_as_array" || options.lengthApi == "tensor_host") {
        Fail("this package uses device Tensor lengths in its canonical API");
    }
#endif
    if (options.benchmarkWarmups < 0 || options.benchmarkRepetitions < 0) {
        Fail("benchmark launch counts must be non-negative");
    }
    if (options.captureWorkspaceErrors &&
        (options.benchmarkRepetitions != 0 || options.benchmarkWarmups != 0)) {
        Fail("--capture-workspace-errors does not support benchmark launches");
    }
    if (!options.workspaceOnly && options.benchmarkRepetitions == 0 &&
        (!std::isfinite(options.rtol) || !std::isfinite(options.atol) ||
         options.rtol < 0.0 || options.atol < 0.0)) {
        Fail("runner requires explicit non-negative --rtol and --atol");
    }
    return options;
}

bool IsWorkspacePreparationError(aclnnStatus status)
{
    // CANN 8.5 aclnn return codes: parameter, shape inference, tiling and
    // kernel lookup failures. Runtime errors (361001), allocation failures
    // and unknown internal errors must still terminate the process.
    return status == 161001 || status == 161002 || status == 561001 ||
           status == 561002 || status == 561003;
}

bool RunCase(const std::filesystem::path &fixtureDir, const RunnerOptions &options,
             aclrtStream stream)
{
    const Config config = ReadConfig(fixtureDir / "case.conf");
    const std::string caseId = Required(config, "case_id");
    if (Required(config, "fixture_format") != "2") {
        Fail(caseId + ": unsupported fixture format");
    }
    const auto requireSelection = [&caseId](const std::string &requested,
                                             const std::string &actual,
                                             const std::string &name) {
        if (!requested.empty() && requested != actual) {
            Fail(caseId + ": requested " + name + "=" + requested +
                 ", fixture has " + actual);
        }
    };
    requireSelection(options.coreDtype, Required(config, "input.query.dtype"), "core dtype");
    requireSelection(options.weightsDtype, Required(config, "input.weights.dtype"), "weights dtype");
    requireSelection(options.dataMode, Required(config, "data_mode"), "data mode");

    std::vector<DeviceTensor> inputs(12);
    std::vector<DeviceTensor> outputs(4);
    using LengthArray = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;
    LengthArray actualQuery(nullptr, aclDestroyIntArray);
    LengthArray actualKey(nullptr, aclDestroyIntArray);
    using TensorMeta = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
    TensorMeta hostQuery(nullptr, aclDestroyTensor), hostKey(nullptr, aclDestroyTensor);
    std::vector<int64_t> hostQueryValues, hostKeyValues;
    for (std::size_t index = 0; index < 12; ++index) {
        const std::string prefix = std::string("input.") + INPUT_NAMES[index] + ".";
        const auto present = config.find(prefix + "present");
        if (present != config.end() && present->second == "0") {
            continue;
        }
        const auto shape = ParseShape(Required(config, prefix + "shape"));
        const auto storageShape = ParseShape(Required(config, prefix + "storage_shape"));
        const auto &dtype = Required(config, prefix + "dtype");
        const auto bytes = TensorBytes(storageShape, dtype);
        const auto host = ReadBinary(fixtureDir / Required(config, prefix + "file"), bytes);
        if (index >= 10 && options.lengthApi == "tensor_host") {
            if (dtype != "int64" || bytes % sizeof(int64_t) != 0) Fail("invalid host length Tensor");
            auto &values = index == 10 ? hostQueryValues : hostKeyValues;
            values.resize(bytes / sizeof(int64_t));
            std::memcpy(values.data(), host.data(), bytes);
            const auto strides = ContiguousStrides(shape);
            auto &tensor = index == 10 ? hostQuery : hostKey;
            tensor.reset(aclCreateTensor(shape.data(), shape.size(), ACL_INT64,
                strides.data(), 0, ACL_FORMAT_ND, storageShape.data(), storageShape.size(), values.data()));
            if (!tensor) Fail("aclCreateTensor host length failed");
            continue;
        }
        if (index >= 10 && options.lengthApi == "array") {
            if (dtype != "int64" || bytes % sizeof(int64_t) != 0) {
                Fail("actual sequence lengths must be int64 arrays");
            }
            std::vector<int64_t> values(bytes / sizeof(int64_t));
            std::memcpy(values.data(), host.data(), bytes);
            auto &array = index == 10 ? actualQuery : actualKey;
            array.reset(aclCreateIntArray(values.data(), values.size()));
            if (!array) Fail("aclCreateIntArray failed");
            continue;
        }
        const auto strideEntry = config.find(prefix + "strides");
        const auto offsetEntry = config.find(prefix + "element_offset");
        inputs[index].Create(shape, dtype, &host, storageShape, 0, false,
            strideEntry == config.end() ? std::vector<int64_t>{} : ParseShape(strideEntry->second),
            offsetEntry == config.end() ? 0 : std::stoll(offsetEntry->second));
    }
    for (std::size_t index = 0; index < 4; ++index) {
        const std::string prefix = std::string("output.") + OUTPUT_NAMES[index] + ".";
        const auto offset = config.find(prefix + "view_offset_bytes");
        outputs[index].Create(ParseShape(Required(config, prefix + "shape")),
                              Required(config, prefix + "dtype"), nullptr, {},
                              offset == config.end() ? 0 : std::stoull(offset->second),
                              offset != config.end());
    }

    const auto keyDim = static_cast<std::size_t>(ParseShape(Required(config, "output.dKeyIndex.shape")).back());
    const auto zeroKeyRows = ReadBinary(fixtureDir / Required(config, "zero_key_rows_file"),
                                       outputs[1].Bytes() / TypeSize(outputs[1].Dtype()) / keyDim);
    const auto scaleValue = std::stod(Required(config, "scale_value"));
    std::string layout = Required(config, "layout");
    const auto sparseMode = std::stoll(Required(config, "sparse_mode"));
    const bool deterministic = std::stoll(Required(config, "deterministic")) != 0;
    Check(aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, deterministic ? 1 : 0),
          caseId + ": set network determinism");
    int64_t networkDeterministic = -1;
    Check(aclrtCtxGetSysParamOpt(ACL_OPT_DETERMINISTIC, &networkDeterministic),
          caseId + ": read network determinism");
    if (networkDeterministic != (deterministic ? 1 : 0)) Fail("determinism setting mismatch");
    std::cout << "case=" << caseId << " network_deterministic=" << networkDeterministic << std::endl;

    std::cout << "case=" << caseId << " length_api=" << options.lengthApi << std::endl;
    const auto queryWorkspace = [&](std::uint64_t *workspace, aclOpExecutor **executor) {
#if SLI_LENGTH_ARRAY_API
        if (options.lengthApi == "tensor" || options.lengthApi == "tensor_host") {
            const auto *queryLengths = options.lengthApi == "tensor_host" ? hostQuery.get() : inputs[10].Get();
            const auto *keyLengths = options.lengthApi == "tensor_host" ? hostKey.get() : inputs[11].Get();
            return aclnnSparseLightningIndexerGradKlLossTensorGetWorkspaceSize(
                inputs[0].Get(), inputs[1].Get(), inputs[2].Get(), inputs[3].Get(),
                inputs[4].Get(), inputs[5].Get(), inputs[6].Get(), inputs[7].Get(),
                inputs[8].Get(), inputs[9].Get(), queryLengths, keyLengths,
                scaleValue, layout.data(), sparseMode, deterministic,
                outputs[0].Get(), outputs[1].Get(), outputs[2].Get(), outputs[3].Get(),
                workspace, executor);
        }
        // This explicit negative control models an outdated caller declaration.
        // It is never selected by normal correctness or benchmark invocations.
        const auto *queryLengths = (options.lengthApi == "tensor_as_array" || options.lengthApi == "canonical_tensor")
            ? reinterpret_cast<const aclIntArray *>(inputs[10].Get()) : actualQuery.get();
        const auto *keyLengths = (options.lengthApi == "tensor_as_array" || options.lengthApi == "canonical_tensor")
            ? reinterpret_cast<const aclIntArray *>(inputs[11].Get()) : actualKey.get();
#else
        const auto *queryLengths = inputs[10].Get();
        const auto *keyLengths = inputs[11].Get();
#endif
        return aclnnSparseLightningIndexerGradKlLossGetWorkspaceSize(
            inputs[0].Get(), inputs[1].Get(), inputs[2].Get(), inputs[3].Get(),
            inputs[4].Get(), inputs[5].Get(), inputs[6].Get(), inputs[7].Get(),
            inputs[8].Get(), inputs[9].Get(), queryLengths, keyLengths,
            scaleValue, layout.data(), sparseMode, deterministic,
            outputs[0].Get(), outputs[1].Get(), outputs[2].Get(), outputs[3].Get(),
            workspace, executor);
    };
    const auto launch = [&](bool timed, float &elapsedMs) {
        std::uint64_t requestedWorkspace = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = queryWorkspace(&requestedWorkspace, &executor);
        Check(status, caseId + ": benchmark workspace query");
        Workspace workspace(requestedWorkspace);
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        if (timed) {
            Check(aclrtCreateEvent(&start), caseId + ": create start event");
            Check(aclrtCreateEvent(&end), caseId + ": create end event");
            Check(aclrtRecordEvent(start, stream), caseId + ": record start event");
        }
        Check(aclnnSparseLightningIndexerGradKlLoss(
                  workspace.Get(), requestedWorkspace, executor, stream),
              caseId + ": benchmark kernel launch");
        if (timed) {
            Check(aclrtRecordEvent(end, stream), caseId + ": record end event");
        }
        Check(aclrtSynchronizeStream(stream), caseId + ": benchmark stream synchronize");
        if (timed) {
            Check(aclrtEventElapsedTime(&elapsedMs, start, end),
                  caseId + ": event elapsed time");
            Check(aclrtDestroyEvent(start), caseId + ": destroy start event");
            Check(aclrtDestroyEvent(end), caseId + ": destroy end event");
        }
        return requestedWorkspace;
    };

    if (options.benchmarkRepetitions > 0) {
        float ignored = 0.0F;
        std::uint64_t workspaceBytes = 0;
        for (int repetition = 0; repetition < options.benchmarkWarmups; ++repetition) {
            workspaceBytes = std::max(workspaceBytes, launch(false, ignored));
        }
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(options.benchmarkRepetitions));
        for (int repetition = 0; repetition < options.benchmarkRepetitions; ++repetition) {
            float elapsedMs = 0.0F;
            workspaceBytes = std::max(workspaceBytes, launch(true, elapsedMs));
            samples.push_back(elapsedMs * 1000.0F);
        }
        std::sort(samples.begin(), samples.end());
        const auto percentile = [&samples](double ratio) {
            const auto index = static_cast<std::size_t>(
                ratio * static_cast<double>(samples.size() - 1));
            return samples[index];
        };
        std::cout << "case=" << caseId << " stage=benchmark"
                  << " warmup=" << options.benchmarkWarmups
                  << " repetitions=" << options.benchmarkRepetitions
                  << " workspace=" << workspaceBytes
                  << " device_us_min=" << std::fixed << std::setprecision(3) << samples.front()
                  << " device_us_p50=" << percentile(0.5)
                  << " device_us_p90=" << percentile(0.9)
                  << " device_us_max=" << samples.back()
                  << " status=PASS" << std::endl;
        return true;
    }

    const int repetitions = std::max(1, std::stoi(Required(config, "repetitions")));
    const bool compare = Required(config, "comparison") == "exact";
    std::vector<std::vector<std::uint8_t>> firstOutputs;
    bool passed = true;
    std::uint64_t workspaceBytes = 0;
    const auto started = std::chrono::steady_clock::now();

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        for (auto &output : outputs) {
            output.ResetGuardedValue();
        }
        std::uint64_t requestedWorkspace = 0;
        aclOpExecutor *executor = nullptr;
        const auto status = queryWorkspace(&requestedWorkspace, &executor);
        if (options.captureWorkspaceErrors && IsWorkspacePreparationError(status)) {
            // Generated ACLNN code assigns the executor before invoking
            // tiling. No second-stage call will release it after a rejection.
            if (executor != nullptr) {
                Check(aclSetAclOpExecutorRepeatable(executor), caseId + ": retain rejected workspace executor");
                Check(aclDestroyAclOpExecutor(executor), caseId + ": destroy rejected workspace executor");
            }
            std::cout << "case=" << caseId << " stage=workspace status=RE"
                      << " aclnn_status=" << status << " kernel_launched=0" << std::endl;
            return false;
        }
        Check(status, caseId + ": workspace query");
        workspaceBytes = std::max(workspaceBytes, requestedWorkspace);
        if (options.workspaceOnly) {
            Check(aclSetAclOpExecutorRepeatable(executor), caseId + ": retain workspace executor");
            Check(aclDestroyAclOpExecutor(executor), caseId + ": destroy workspace executor");
            std::cout << "case=" << caseId << " stage=workspace workspace="
                      << requestedWorkspace << " status=PASS" << std::endl;
            return true;
        }
        Workspace workspace(requestedWorkspace);
        aclrtEvent beginEvent = nullptr, endEvent = nullptr;
        Check(aclrtCreateEvent(&beginEvent), caseId + ": create timing event");
        Check(aclrtCreateEvent(&endEvent), caseId + ": create timing event");
        Check(aclrtRecordEvent(beginEvent, stream), caseId + ": start timing");
        Check(aclnnSparseLightningIndexerGradKlLoss(
                  workspace.Get(), requestedWorkspace, executor, stream),
              caseId + ": kernel launch");
        Check(aclrtRecordEvent(endEvent, stream), caseId + ": end timing");
        Check(aclrtSynchronizeStream(stream), caseId + ": stream synchronize");
        float deviceMs = 0.0F;
        Check(aclrtEventElapsedTime(&deviceMs, beginEvent, endEvent), caseId + ": elapsed time");
        Check(aclrtDestroyEvent(beginEvent), caseId + ": destroy timing event");
        Check(aclrtDestroyEvent(endEvent), caseId + ": destroy timing event");
        std::cout << "case=" << caseId << " repetition=" << repetition
                  << " kernel_device_us=" << std::setprecision(9) << deviceMs * 1000.0F
                  << " warmup=" << repetition << std::endl;

        std::vector<std::vector<std::uint8_t>> observed;
        observed.reserve(4);
        for (const auto &output : outputs) {
            observed.emplace_back(output.CopyToHost());
        }
        if (repetition == 0) {
            firstOutputs = observed;
        } else if (deterministic && observed != firstOutputs) {
            std::cerr << "case=" << caseId << " repetition=" << repetition
                      << " deterministic_output=FAIL\n";
            passed = false;
        }
        if (compare) {
            bool zeroPassed = true;
            std::size_t zeroCount = 0;
            for (std::size_t row = 0; row < zeroKeyRows.size() && zeroPassed; ++row) {
                if (!zeroKeyRows[row]) continue;
                ++zeroCount;
                for (std::size_t d = 0; d < keyDim; ++d) {
                    if (Decode(observed[1].data(), row * keyDim + d, outputs[1].Dtype()) != 0.0) {
                        zeroPassed = false;
                        break;
                    }
                }
            }
            passed = passed && zeroPassed;
            std::cout << "case=" << caseId << " unselected_key_rows_zero="
                      << (zeroPassed ? "PASS" : "FAIL") << " checked_rows=" << zeroCount << std::endl;
            for (std::size_t index = 0; index < 4; ++index) {
                const std::string prefix = std::string("output.") + OUTPUT_NAMES[index] + ".";
                const auto golden = ReadBinary(
                    fixtureDir / Required(config, prefix + "golden"), outputs[index].Bytes());
                const auto metrics = Compare(observed[index], golden, outputs[index].Dtype(),
                                             options.rtol, options.atol);
                passed = passed && metrics.passed;
                std::cout << "case=" << caseId << " output=" << OUTPUT_NAMES[index]
                          << " repetition=" << repetition
                          << " status=" << (metrics.passed ? "PASS" : "FAIL")
                          << " max_abs=" << std::scientific << std::setprecision(9) << metrics.maxAbsolute
                          << " max_rel=" << metrics.maxRelative
                          << " error_count=" << metrics.errorCount
                          << " element_count=" << metrics.elementCount
                          << " error_ratio_percent=" << (100.0 * metrics.errorCount / metrics.elementCount);
                if (metrics.firstErrorIndex != std::numeric_limits<std::size_t>::max()) {
                    std::cout << std::setprecision(17)
                              << " first_error_index=" << metrics.firstErrorIndex
                              << " first_error_actual=" << metrics.firstErrorActual
                              << " first_error_golden=" << metrics.firstErrorGolden;
                }
                std::cout << '\n';
                if (options.dumpValues > 0) {
                    const std::string dtype = outputs[index].Dtype();
                    const std::size_t count = std::min<std::size_t>(
                        static_cast<std::size_t>(options.dumpValues),
                        observed[index].size() / TypeSize(dtype));
                    std::cout << "dump=" << OUTPUT_NAMES[index];
                    for (std::size_t item = 0; item < count; ++item) {
                        std::cout << ' ' << std::setprecision(9)
                                  << Decode(observed[index].data(), item, dtype) << '/'
                                  << Decode(golden.data(), item, dtype);
                    }
                    std::cout << '\n';
                }
            }
        }
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "case=" << caseId << " comparison=" << (compare ? "exact" : "launch_only")
              << " repetitions=" << repetitions << " workspace=" << workspaceBytes
              << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsed
              << " status=" << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto options = ParseArguments(argc, argv);
        if (options.showHelp) {
            PrintUsage(std::cout);
            return 0;
        }
        AclSession session;
        VerifyLoadedOpApi();
        session.Initialize(options.deviceId);
        bool passed = true;
        for (const auto &fixture : options.fixtureDirs) {
            if (!RunCase(fixture, options, session.Stream())) {
                passed = false;
                if (!options.captureWorkspaceErrors && !options.continueComparisonErrors) {
                    return 1;
                }
            }
        }
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
