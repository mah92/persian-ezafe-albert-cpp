#include <iostream>
#include "ezafe_detector.hpp"

int main() {
    EzafeDetector detector(
        "model/model_quantized.onnx",
        "model/spiece.model"
    );

    std::vector<std::string> words = {
        "کتابخانه", "مرکزی", "دانشگاه", "شریف"
    };

    auto results = detector.predict(words);

    for (const auto& r : results) {
        std::cout << r.word << " | "
                  << (r.needs_ezafe ? "NEEDS_EZAFE" : "NO_EZAFE")
                  << " | confidence=" << r.confidence << "\n";
    }

    return 0;
}