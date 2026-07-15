#pragma once

#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "sentencepiece_minimal.h"

class EzafeDetector {
public:
    struct Result {
        std::string word;
        bool needs_ezafe;
        float confidence;
    };

    /** @param onnx_path  Path to model_quantized.onnx
     *  @param spiece_path Path to spiece.model  */
    EzafeDetector(const std::string& onnx_path, const std::string& spiece_path);

    /** Predict ezafe for list of space-separated Persian words. */
    std::vector<Result> predict(const std::vector<std::string>& words);

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo mem_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    sentencepiece_minimal::SentencePieceProcessor tokenizer_;

    static constexpr int64_t kMaxLength = 128;
    static constexpr int64_t kPadId = 0;
    static constexpr int64_t kClsId = 2;
    static constexpr int64_t kSepId = 3;
};
