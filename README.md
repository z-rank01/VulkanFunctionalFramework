# Vulkan Functional Framework
面向不同图形 API、以函数式编程（Functional Programming）风格构建的个人定制 Vulkan 框架

# Usage（使用）
## 配置 CMakePresets.json
为了编译源码，需要修改 CMakePresets.json 中的多处路径。
1. 选择一个构建链（build chain），例如名称为 `windows-msvc` 或 `linux-clang`
2. 在 `cacheVariables` 部分修改编译器路径
    - `CMAKE_C_COMPILER`
    - `CMAKE_CXX_COMPILER`
    - `CMAKE_RC_COMPILER`
    - ...

## 配置 app_config.json

## Clangd 相关
如果需要使用 clangd 作为（此处补充说明：例如语言服务器/代码补全等）

## Vulkan SDK
要使用该框架，需要下载并安装 Vulkan SDK，并将其加入系统环境变量。

---

# Vulkan Functional Framework
Personal customized vulkan framework targeting on different gfx APIs in a style of **Functional Progamming**

# Usage
## Config CMakePresets.json
In order to build source files, one need to modify multiple path inside CMakePresets.json. 
1. Target one of the build chain, like name `windows-msvc` or `linux-clang`
2. Modify the path to the compiler in the part of `cacheVariables`
    - `CMAKE_C_COMPILER`
    - `CMAKE_CXX_COMPILER`
    - `CMAKE_RC_COMPILER`
    - ...

## Config app_config.json

## Clangd Specified
If one want to use clangd as 

## Vulkan SDK
To use this framework, one need to download and install vulkan sdk with system variables added.