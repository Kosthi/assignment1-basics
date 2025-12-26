#pragma once

#include <string>
#include <vector>
#include <map>

namespace bpe {
    using Bytes = std::string;

    struct Result {
        std::map<int, Bytes> vocab;
        std::vector<std::pair<Bytes, Bytes> > merges;
    };

    Result train(
        const std::vector<std::string> &distinct_words,
        const std::vector<int> &counts,
        int vocab_size,
        const std::vector<std::string> &special_tokens
    );
}
