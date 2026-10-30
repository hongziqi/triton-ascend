#ifndef ASCEND_PLUGIN_ARG_PARSER_H
#define ASCEND_PLUGIN_ARG_PARSER_H

#include "llvm/ADT/StringRef.h"
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace ascend_pass_plugin {

inline std::optional<std::string>
findArg(const std::vector<std::string> &args, llvm::StringRef key) {
  for (const auto &entry : args) {
    auto pos = entry.find('=');
    if (pos == std::string::npos)
      continue;
    if (entry.substr(0, pos) == key)
      return entry.substr(pos + 1);
  }
  return std::nullopt;
}

inline bool parseBool(const std::vector<std::string> &args, llvm::StringRef key,
                      bool defaultValue) {
  if (auto v = findArg(args, key)) {
    if (*v == "1" || *v == "true" || *v == "True")
      return true;
    if (*v == "0" || *v == "false" || *v == "False")
      return false;
  }
  return defaultValue;
}

inline int parseInt(const std::vector<std::string> &args, llvm::StringRef key,
                    int defaultValue) {
  if (auto v = findArg(args, key))
    return std::atoi(v->c_str());
  return defaultValue;
}

inline std::uint64_t parseUInt64(const std::vector<std::string> &args,
                                 llvm::StringRef key,
                                 std::uint64_t defaultValue) {
  if (auto v = findArg(args, key))
    return std::strtoull(v->c_str(), nullptr, 10);
  return defaultValue;
}

inline std::string parseString(const std::vector<std::string> &args,
                               llvm::StringRef key,
                               const std::string &defaultValue) {
  if (auto v = findArg(args, key))
    return *v;
  return defaultValue;
}

} // namespace ascend_pass_plugin

#endif
