#include "NsfwDetector.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <thread>
#include <memory>
#include <android/log.h>
#include "onnxruntime_cxx_api.h"

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

struct NsfwDetectorImpl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    bool isInitialized = false;
    std::mutex mutex;

    const int INPUT_WIDTH = 320;
    const int INPUT_HEIGHT = 320;

    const std::vector<int> CENSORED_CLASS_IDS = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17};

    void init() {
        std::lock_guard<std::mutex> lock(mutex);
        if (isInitialized) return;

        try {
            ALOGD("init(): Creating Ort::Env...");
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NsfwDetector");
            ALOGD("init(): Ort::Env created successfully");

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

            const char* modelPath = "/system/etc/nsfw_model.onnx";
            ALOGD("init(): Checking model file at %s", modelPath);
            std::ifstream f(modelPath);
            if (!f.good()) {
                ALOGE("Model file not found at %s", modelPath);
                return;
            }
            ALOGD("init(): Model file exists, creating session...");

            session = std::make_unique<Ort::Session>(*env, modelPath, sessionOptions);
            isInitialized = true;
            ALOGD("NsfwDetector initialized successfully");
        } catch (const Ort::Exception& e) {
            ALOGE("ONNX Runtime initialization error: %s", e.what());
        } catch (const std::exception& e) {
            ALOGE("Failed to initialize: %s", e.what());
        } catch (...) {
            ALOGE("Unknown exception during initialization");
        }
    }
};

NsfwDetector& NsfwDetector::getInstance() {
    static NsfwDetector instance;
    return instance;
}

NsfwDetector::NsfwDetector() : pImpl(std::make_unique<NsfwDetectorImpl>()) {
}

NsfwDetector::~NsfwDetector() = default;

static std::vector<float> preprocess_helper(const uint8_t* pixels, int width, int height, int stride, int target_w, int target_h) {
    float scale = std::min((float)target_w / width, (float)target_h / height);
    int new_w = (int)(width * scale);
    int new_h = (int)(height * scale);
    
    std::vector<float> input_tensor_values(target_w * target_h * 3, 0.0f);

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

            int pixel_idx = y * target_w + x;
            
            input_tensor_values[pixel_idx] = r / 255.0f;
            input_tensor_values[target_w * target_h + pixel_idx] = g / 255.0f;
            input_tensor_values[2 * target_w * target_h + pixel_idx] = b / 255.0f;
        }
    }
    return input_tensor_values;
}

std::optional<std::vector<Detection>> NsfwDetector::detect(const uint8_t* pixels, int width, int height, int stride, uint32_t generationId) {
    ALOGD("detect() CALLED: genId=%u, size=%dx%d, isInit=%d", generationId, width, height, pImpl->isInitialized);

    if (!pImpl->isInitialized) {
        ALOGD("detect(): Calling init()...");
        pImpl->init();
        if (!pImpl->isInitialized) {
            ALOGE("detect(): Init FAILED! Returning empty detections");
            return std::vector<Detection>();
        }
        ALOGD("detect(): Init SUCCESS!");
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(generationId);
        if (it != cache_.end()) return it->second;
        if (processing_[generationId]) return std::nullopt;
        processing_[generationId] = true;
    }

    std::vector<float> input_values = preprocess_helper(pixels, width, height, stride, pImpl->INPUT_WIDTH, pImpl->INPUT_HEIGHT);

    std::thread([this, input_values, width, height, generationId]() {
        runInferenceAsync(input_values, width, height, generationId);
    }).detach();

    return std::nullopt;
}

void NsfwDetector::runInferenceAsync(std::vector<float> input_tensor_values, int width, int height, uint32_t generationId) {
    ALOGD("runInferenceAsync: START genId=%u, size=%dx%d, input_size=%zu",
          generationId, width, height, input_tensor_values.size());

    if (!pImpl->isInitialized || !pImpl->session) {
        ALOGE("runInferenceAsync: Session not initialized! genId=%u", generationId);
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
        return;
    }

    int INPUT_WIDTH = pImpl->INPUT_WIDTH;
    int INPUT_HEIGHT = pImpl->INPUT_HEIGHT;

    // Inference
    std::vector<int64_t> input_node_dims = {1, 3, INPUT_WIDTH, INPUT_HEIGHT};

    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_node_dims.data(), input_node_dims.size());

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};

        ALOGD("runInferenceAsync: Before ONNX Run genId=%u", generationId);
        std::vector<Ort::Value> output_tensors = pImpl->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        ALOGD("runInferenceAsync: After ONNX Run genId=%u", generationId);

        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        int channels = shape[1]; 
        int anchors = shape[2];
        
        std::vector<Detection> detections;
        
        float scale = std::min((float)INPUT_WIDTH / width, (float)INPUT_HEIGHT / height);
        int new_w = (int)(width * scale);
        int new_h = (int)(height * scale);

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
                for (int id : pImpl->CENSORED_CLASS_IDS) {
                    if (id == max_class_id) {
                        is_censored = true;
                        break;
                    }
                }
                
                if (is_censored) {
                    ALOGD("Found NSFW item: Class %d with score %.2f", max_class_id, max_score);

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

        ALOGD("runInferenceAsync: Processing complete, found %zu detections for genId=%u",
              detections.size(), generationId);

        std::lock_guard<std::mutex> lock(cacheMutex_);
        if (cache_.size() > 500) {
            ALOGD("runInferenceAsync: Cache full, clearing...");
            cache_.clear();
        }
        cache_[generationId] = detections;
        processing_.erase(generationId);
        ALOGD("runInferenceAsync: Results cached for genId=%u", generationId);

    } catch (const Ort::Exception& e) {
        ALOGE("ONNX Runtime error for genId=%u: %s", generationId, e.what());
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
    } catch (const std::exception& e) {
        ALOGE("Async Inference failed for genId=%u: %s", generationId, e.what());
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
    } catch (...) {
        ALOGE("Unknown exception in runInferenceAsync for genId=%u", generationId);
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[generationId] = {};
        processing_.erase(generationId);
    }
}

} // namespace uirenderer
} // namespace android