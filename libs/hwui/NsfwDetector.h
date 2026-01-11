#pragma once

#include <android/log.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <thread>
#include "onnxruntime_cxx_api.h"

// Log tag
#define LOG_TAG "NsfwDetector"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace android {
namespace uirenderer {

struct Detection {
    float x, y, w, h;
    float score;
    int classId;
};

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
    ~NsfwDetector() = default;

    // Model yükleme ve hazırlık
    void init();
    
    // Arka plan işlemi
    void runInferenceAsync(std::vector<float> input_values, int original_w, int original_h, uint32_t generationId);

    // Yardımcı fonksiyonlar
    std::vector<float> preprocess(const uint8_t* pixels, int width, int height, int stride, float& x_ratio, float& y_ratio, float& x_pad, float& y_pad);

    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
    bool isInitialized_ = false;
    std::mutex mutex_;
    
    // Cache
    // Key: Bitmap Generation ID
    // Value: Detection listesi
    std::mutex cacheMutex_;
    std::unordered_map<uint32_t, std::vector<Detection>> cache_;
    // İşlenmekte olanlar listesi (aynı resmi 10 kere thread'e atmayalım)
    std::unordered_map<uint32_t, bool> processing_;
    
    // Model input size (NudeNet 320n)
    const int INPUT_WIDTH = 320;
    const int INPUT_HEIGHT = 320;
    
    // Sınıflar
    const std::vector<std::string> LABELS = {
        "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
        "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
        "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED"
    };

    // Sansürlenecek sınıfların ID'leri: Yüzler hariç hepsi
    // 1: FACE_FEMALE ve 12: FACE_MALE hariç tutuldu.
    const std::vector<int> CENSORED_CLASS_IDS = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17};
};

} // namespace uirenderer
} // namespace android
