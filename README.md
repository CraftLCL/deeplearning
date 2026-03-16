# 课程内容发布说明

本仓库用于深度学习课程讲解，当前主要内容为各章节 Notebook：

- `chapter1.ipynb`
- `chapter2.ipynb`
- `chapter3.ipynb`
- `chapter4.ipynb`
- `chapter5.ipynb`
- `chapter6.ipynb`
- `chapter7.ipynb`
- `chapter1.md`
- `photos/`

## 发布前检查

对外分享或提交课程内容前，请确认以下内容：

1. 课堂要用到的 Notebook 都能正常打开。
2. 关键代码单元已运行，输出结果符合讲课预期。
3. 删除无关的临时文件（如测试截图、缓存文件等）。
4. 文件命名保持统一规范，避免课堂中链接失效。

## 发布方式（压缩包）

推荐将整个项目目录打包后上传到课程平台或分享给学员。

### 1. 进入项目上级目录

```bash
cd /home/gawainli/project
```

### 2. 创建压缩包

```bash
tar -czf deeplearning_course_materials.tar.gz deeplearning
```

### 3. 检查压缩包内容

```bash
tar -tzf deeplearning_course_materials.tar.gz
```

确认包含所有需要的 `.ipynb` 文件以及图片目录后再发布。

## 发布方式（Git，可选）

如果你通过 Git 仓库维护讲课内容，可使用：

```bash
cd /home/gawainli/project/deeplearning
git add .
git commit -m "更新"
git push
```

## 建议

- 每次讲课前先打一个发布包，确保版本可追溯。
- 保留一份本地备份（压缩包或 Git tag）以防同步失败。
