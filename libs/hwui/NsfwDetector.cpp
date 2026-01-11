#include "NsfwDetector.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <thread>
#include <android/log.h>

// GARANTİ LOG MAKROLARI
// Eğer sistemde ALOGE tanımlıysa onu kullan, yoksa kendimiz tanımlayalım.
// Ama önce LOG_TAG'ı ayarlayalım.
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "NsfwDetector"

#ifndef ALOGE
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif

#ifndef ALOGD
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#endif

namespace android {
namespace uirenderer {

NsfwDetector& NsfwDetector::getInstance() {
    static NsfwDetector instance;
    return instance;
}

NsfwDetector::NsfwDetector() {
    // init() lazy loading
}

void NsfwDetector::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (isInitialized_) return;

    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NsfwDetector");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        const char* modelPath = "/system/etc/nsfw_model.onnx"; 
        
        std::ifstream f(modelPath);
        if (!f.good()) {
            ALOGE("Model file not found at %s", modelPath);
            return;
        }

        session_ = Ort::Session(env_, modelPath, sessionOptions);
        isInitialized_ = true;
        ALOGD("NsfwDetector initialized successfully");
    } catch (const Ort::Exception& e) {
        ALOGE("Failed to initialize NsfwDetector: %s", e.what());
    } catch (const std::exception& e) {
        ALOGE("Exception in init: %s", e.what());
    }
}

std::vector<float> NsfwDetector::preprocess(const uint8_t* pixels, int width, int height, int stride, 
                                          float& x_ratio, float& y_ratio, float& x_pad, float& y_pad) {
    
    float scale = std::min((float)INPUT_WIDTH / width, (float)INPUT_HEIGHT / height);
    int new_w = (int)(width * scale);
    int new_h = (int)(height * scale);
    
    x_ratio = (float)INPUT_WIDTH / new_w;
    y_ratio = (float)INPUT_HEIGHT / new_h;
    
    x_pad = INPUT_WIDTH - new_w;
    y_pad = INPUT_HEIGHT - new_h;
    
    std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3, 0.0f);

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_x = (int)(x / scale);
            int src_y = (int)(y / scale);
            
            if (src_x >= width) src_x = width - 1;
            if (src_y >= height) src_y = height - 1;

            const uint8_t* pixel = pixels + (src_y * stride) + (src_x * 4);
            uint8_t r = pixel[0];
            uint8_t g = pixel[1];
            uint8_t b = pixel[2];

            int pixel_idx = y * INPUT_WIDTH + x;
            
            input_tensor_values[pixel_idx] = r / 255.0f;
            input_tensor_values[INPUT_WIDTH * INPUT_HEIGHT + pixel_idx] = g / 255.0f;
            input_tensor_values[2 * INPUT_WIDTH * INPUT_HEIGHT + pixel_idx] = b / 255.0f;
        }
    }
    
    return input_tensor_values;
}

std::optional<std::vector<Detection>> NsfwDetector::detect(const uint8_t* pixels, int width, int height, int stride, uint32_t generationId) {
    if (!isInitialized_) {
        init();
        if (!isInitialized_) return std::vector<Detection>(); 
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(generationId);
        if (it != cache_.end()) {
            return it->second; 
        }

        if (processing_[generationId]) {
            return std::nullopt; 
        }
        
        processing_[generationId] = true;
    }

    float x_ratio, y_ratio, x_pad, y_pad;
    std::vector<float> input_values = preprocess(pixels, width, height, stride, x_ratio, y_ratio, x_pad, y_pad);

    std::thread([this, input_values, width, height, generationId]() {
        runInferenceAsync(input_values, width, height, generationId);
    }).detach();

    return std::nullopt;
}

void NsfwDetector::runInferenceAsync(std::vector<float> input_tensor_values, int width, int height, uint32_t generationId) {
    float scale = std::min((float)INPUT_WIDTH / width, (float)INPUT_HEIGHT / height);
    int new_w = (int)(width * scale);
    int new_h = (int)(height * scale);

    std::vector<int64_t> input_node_dims = {1, 3, INPUT_WIDTH, INPUT_HEIGHT};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_node_dims.data(), input_node_dims.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    } catch (const Ort::Exception& e) {
        ALOGE("Async Inference failed: %s", e.what());
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
        return;
    } catch (const std::exception& e) {
        ALOGE("Async Inference std exception: %s", e.what());
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
        return;
    }

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    int channels = shape[1]; 
    int anchors = shape[2];
    
    std::vector<Detection> detections;

    for (int i = 0; i < anchors; ++i) {
        float max_score = 0;
        int max_class_id = -1;
        
        for (int c = 4; c < channels; ++c) {
            float score = floatarr[c * anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = c - 4;
            }
        }

        if (max_score >= 0.2f) {
            bool is_censored = false;
            for (int id : CENSORED_CLASS_IDS) {
                if (id == max_class_id) {
                    is_censored = true;
                    break;
                }
            }
            
            if (is_censored) {
                float x = floatarr[0 * anchors + i];
                float y = floatarr[1 * anchors + i];
                float w = floatarr[2 * anchors + i];
                float h = floatarr[3 * anchors + i];
                
                float x1 = x - w / 2;
                float y1 = y - h / 2;
                
                x1 = (x1 * INPUT_WIDTH) / new_w;
                y1 = (y1 * INPUT_HEIGHT) / new_h;
                w = (w * INPUT_WIDTH) / new_w;
                h = (h * INPUT_HEIGHT) / new_h;
                
                float x_final = x1 * (width / (float)new_w);
                float y_final = y1 * (height / (float)new_h);
                float w_final = w * (width / (float)new_w);
                float h_final = h * (height / (float)new_h);

                detections.push_back({x_final, y_final, w_final, h_final, max_score, max_class_id});
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        if (cache_.size() > 500) {
            cache_.clear();
        }
        cache_[generationId] = detections;
        processing_.erase(generationId); 
    }
}

} // namespace uirenderer
} // namespace android
