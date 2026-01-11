#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <memory>

namespace android {
namespace uirenderer {

struct Detection {
    float x, y, w, h;
    float score;
    int classId;
};

// Forward declaration for Pimpl
struct NsfwDetectorImpl;

class NsfwDetector {
public:
    static NsfwDetector& getInstance();

    // Resmi analiz et ve NSFW bölgelerini döndür
    // generationId: Bitmap'in benzersiz kimliği
    // Return:
    //   std::nullopt -> Henüz işleniyor (Beklemede) -> Yeşil Kutu Çiz
    //   std::vector  -> İşlendi -> Sonuca göre çiz
    std::optional<std::vector<Detection>> detect(const uint8_t* pixels, int width, int height, int stride, uint32_t generationId);

private:
    NsfwDetector();
    ~NsfwDetector();

    // Arka plan işlemi
    void runInferenceAsync(std::vector<float> input_values, int original_w, int original_h, uint32_t generationId);

    // Implementation details (ONNX objects hidden here)
    std::unique_ptr<NsfwDetectorImpl> pImpl;
    
    // Cache
    std::mutex cacheMutex_;
    std::unordered_map<uint32_t, std::vector<Detection>> cache_;
    std::unordered_map<uint32_t, bool> processing_;
};

} // namespace uirenderer
} // namespace android
