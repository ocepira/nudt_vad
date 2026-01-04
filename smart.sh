#!/bin/bash
# 自动添加 <100MB 的文件

# 获取所有未追踪文件
git ls-files --others --exclude-standard | while read file; do
  if [ -f "$file" ]; then
    size=$(du -b "$file" | cut -f1)
    
    # 100MB = 104857600 字节
    if [ "$size" -gt 104857600 ]; then
      echo "跳过大文件: $file (${size} bytes)"
    else
      git add "$file"
      echo "✅ 已添加: $file"
    fi
  fi
done

# 查看暂存区
git status