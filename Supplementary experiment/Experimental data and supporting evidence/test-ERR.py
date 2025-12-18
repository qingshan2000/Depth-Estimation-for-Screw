import os
import cv2
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as T
import json

from utils.depth_model import load_depth_model
from models.screw_regressor import ScrewRegressor

# ==== 硬编码配置 ====
ENCODER = 'vitl'
DEPTH_CKPT = 'checkpoints/depth_anything_v2_vitl.pth'
INPUT_SIZE = 518

REG_CKPT = 'checkpoints/best_screw_regressor1207.pth'

TEST_IMAGES = 'data/test/images1204'
TEST_CACHE = 'data/test/cache1204'
TEST_MASKS = 'data/test/masks1204'

OUTPUT_CSV = 'results/predictions_circle_offset1207.csv'
LABEL_CSV = 'data/test/label1204.csv'  # 标签文件路径
ERROR_STATS_CSV = 'results/表9_螺栓到表面距离预测误差统计.csv'  # 表9输出路径
ERROR_REPORT_JSON = 'results/error_report.json'  # 误差报告JSON
ERROR_REPORT_TXT = 'results/error_report.txt'  # 误差报告文本

TRANSFORM_SIZE = 224

# 支持的图像格式
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'}


def is_image_file(fname):
    return os.path.splitext(fname)[-1] in IMAGE_EXTS


def safe_compute_avg_depth(depth_map, mask):
    """安全的平均深度计算"""
    if depth_map is None or mask is None:
        return 0.0

    if not mask.any():
        return 0.0

    masked_depth = depth_map[mask]

    # 检查数据有效性
    if len(masked_depth) == 0:
        return 0.0

    if np.isnan(masked_depth).any() or np.isinf(masked_depth).any():
        masked_depth = np.nan_to_num(masked_depth, nan=0.0, posinf=1000.0, neginf=0.0)

    # 移除极端值
    if len(masked_depth) > 10:
        lower = np.percentile(masked_depth, 5)
        upper = np.percentile(masked_depth, 95)
        masked_depth = masked_depth[(masked_depth >= lower) & (masked_depth <= upper)]

    if len(masked_depth) == 0:
        return 0.0

    return np.mean(masked_depth)


def check_cache_files(cache_dir, masks_dir, image_name, rel_dir):
    """检查缓存文件是否存在"""
    # 深度图缓存路径
    depth_cache_file = os.path.join(
        cache_dir,
        rel_dir,
        f"{image_name}.npy"
    )

    # 掩码缓存路径
    inner_mask_file = os.path.join(
        masks_dir,
        rel_dir,
        f"{image_name}_inner_mask.png"
    )

    outer_mask_file = os.path.join(
        masks_dir,
        rel_dir,
        f"{image_name}_outer_mask.png"
    )

    # 检查所有文件是否存在
    depth_exists = os.path.exists(depth_cache_file)
    inner_exists = os.path.exists(inner_mask_file)
    outer_exists = os.path.exists(outer_mask_file)

    return depth_exists and inner_exists and outer_exists, depth_cache_file, inner_mask_file, outer_mask_file


def load_cached_data(depth_cache_file, inner_mask_file, outer_mask_file, target_size=None):
    """从缓存加载深度图和掩码"""
    try:
        # 加载深度图
        depth = np.load(depth_cache_file)

        # 加载掩码
        inner_mask = cv2.imread(inner_mask_file, cv2.IMREAD_GRAYSCALE)
        outer_mask = cv2.imread(outer_mask_file, cv2.IMREAD_GRAYSCALE)

        if inner_mask is None or outer_mask is None:
            raise ValueError("无法加载掩码文件")

        # 如果需要调整尺寸
        if target_size is not None:
            h, w = target_size
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            if inner_mask.shape != (h, w):
                inner_mask = cv2.resize(inner_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if outer_mask.shape != (h, w):
                outer_mask = cv2.resize(outer_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return depth, inner_mask, outer_mask

    except Exception as e:
        raise ValueError(f"加载缓存数据失败: {str(e)}")


def analyze_prediction_errors(all_distance_info, output_path):
    """
    分析预测误差并生成表9格式的统计

    参数:
        all_distance_info: 包含真实距离和预测距离的列表
        output_path: 输出文件路径
    """
    if not all_distance_info:
        print("警告：没有可用的距离数据进行误差分析")
        return None

    # 转换为DataFrame
    df_errors = pd.DataFrame(all_distance_info)

    # 计算绝对误差
    df_errors['absolute_error'] = df_errors['error'].abs()

    # 计算统计指标
    absolute_errors = df_errors['absolute_error'].values

    # 1. 预测平均绝对误差 (MAE)
    mae = np.mean(absolute_errors)

    # 2. 预测误差标准差
    std_error = np.std(absolute_errors)

    # 3. 95% 预测误差区间
    error_95_lower = mae - 1.96 * std_error
    error_95_upper = mae + 1.96 * std_error

    # 4. RMSE
    rmse = np.sqrt(np.mean(np.square(df_errors['error'].values)))

    # 5. 相对误差均值
    relative_errors = []
    for _, row in df_errors.iterrows():
        if row['true_distance'] > 0:
            rel_error = abs(row['error'] / row['true_distance']) * 100
            relative_errors.append(rel_error)
    mean_relative_error = np.mean(relative_errors) if relative_errors else 0

    # 创建表9格式的统计
    table9_data = [
        {
            '统计指标': '预测平均绝对误差 (MAE)',
            '计算公式': r'\(\frac{1}{M} \sum_{j=1}^{M} | y_j - \hat{y}_j |\)',
            '数值 (mm)': f"{mae:.2f}"
        },
        {
            '统计指标': '预测误差标准差',
            '计算公式': r'\(\sqrt{\frac{1}{M-1} \sum_{j=1}^{M} (|y_j - \hat{y}_j| - MAE)^2}\)',
            '数值 (mm)': f"{std_error:.2f}"
        },
        {
            '统计指标': '95% 预测误差区间下限',
            '计算公式': 'MAE - 1.96 × Std',
            '数值 (mm)': f"{error_95_lower:.2f}"
        },
        {
            '统计指标': '95% 预测误差区间上限',
            '计算公式': 'MAE + 1.96 × Std',
            '数值 (mm)': f"{error_95_upper:.2f}"
        },
        {
            '统计指标': 'RMSE',
            '计算公式': r'\(\sqrt{\frac{1}{M} \sum_{j=1}^{M} (y_j - \hat{y}_j)^2}\)',
            '数值 (mm)': f"{rmse:.2f}"
        },
        {
            '统计指标': '相对误差均值',
            '计算公式': r'\(\frac{1}{M} \sum_{j=1}^{M} \frac{|y_j - \hat{y}_j|}{y_j} \times 100\%\)',
            '数值 (%)': f"{mean_relative_error:.1f}"
        }
    ]

    table9_df = pd.DataFrame(table9_data)

    # 保存表9
    table9_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 生成完整统计报告
    full_stats = {
        '总样本数': len(df_errors),
        '预测平均绝对误差(MAE)': float(mae),
        '预测误差标准差': float(std_error),
        '95%预测误差区间': [float(error_95_lower), float(error_95_upper)],
        'RMSE': float(rmse),
        '相对误差均值(%)': float(mean_relative_error),
        '最大绝对误差': float(np.max(absolute_errors)),
        '最小绝对误差': float(np.min(absolute_errors)),
        '平均预测值': float(np.mean(df_errors['pred_distance'].values)),
        '平均真实值': float(np.mean(df_errors['true_distance'].values)),
    }

    # 保存JSON报告
    with open(ERROR_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)

    # 生成文本报告
    with open(ERROR_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("螺栓到表面距离预测误差统计（测试集）\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 总体统计\n")
        f.write(f"   总样本数: {full_stats['总样本数']}个\n")
        f.write(f"   平均预测值: {full_stats['平均预测值']:.2f} mm\n")
        f.write(f"   平均真实值: {full_stats['平均真实值']:.2f} mm\n\n")

        f.write("2. 误差统计指标\n")
        f.write(f"   预测平均绝对误差(MAE): {full_stats['预测平均绝对误差(MAE)']:.2f} mm\n")
        f.write(f"   预测误差标准差: {full_stats['预测误差标准差']:.2f} mm\n")
        f.write(
            f"   95%预测误差区间: [{full_stats['95%预测误差区间'][0]:.2f}, {full_stats['95%预测误差区间'][1]:.2f}] mm\n")
        f.write(f"   RMSE: {full_stats['RMSE']:.2f} mm\n")
        f.write(f"   相对误差均值: {full_stats['相对误差均值(%)']:.1f} %\n")
        f.write(f"   最大绝对误差: {full_stats['最大绝对误差']:.2f} mm\n")
        f.write(f"   最小绝对误差: {full_stats['最小绝对误差']:.2f} mm\n")

    print(f"\n误差分析完成!")
    print(f"表9已保存到: {output_path}")
    print(f"完整统计报告已保存到: {ERROR_REPORT_JSON}")
    print(f"文本报告已保存到: {ERROR_REPORT_TXT}")

    return full_stats, table9_df


def load_labels(label_file):
    """加载标签文件，只需要距离信息"""
    if not os.path.exists(label_file):
        print(f"警告：标签文件 {label_file} 不存在，将只进行预测，不计算误差")
        return None

    try:
        labels_df = pd.read_csv(label_file)
        print(f"成功加载标签文件，共 {len(labels_df)} 条记录")

        # 去除列名中的空格
        labels_df.columns = labels_df.columns.str.strip()

        # 检查必需的列
        required_columns = ['image', 'distance']
        for col in required_columns:
            if col not in labels_df.columns:
                print(f"错误: 标签文件缺少必需的列 '{col}'")
                return None

        # 创建标签字典
        label_dict = {}
        for _, row in labels_df.iterrows():
            image_name = str(row['image'])
            # 去除扩展名
            base_name = os.path.splitext(image_name)[0]
            try:
                distance = float(row['distance'])
                label_dict[base_name] = distance
            except (ValueError, TypeError):
                print(f"警告：图像 {image_name} 的距离值格式错误: {row['distance']}")
                continue

        print(f"已加载 {len(label_dict)} 张图像的距离标签")

        # 输出距离统计信息
        if label_dict:
            distances = list(label_dict.values())
            print(f"距离范围: [{min(distances):.2f}, {max(distances):.2f}] mm")
            print(f"平均距离: {np.mean(distances):.2f} ± {np.std(distances):.2f} mm")

        return label_dict

    except Exception as e:
        print(f"加载标签文件时出错: {e}")
        return None


def main():
    # 1. 创建必要目录
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 加载标签
    label_dict = load_labels(LABEL_CSV)
    has_labels = label_dict is not None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 加载回归模型
    print("Loading regressor model...")
    reg_model = ScrewRegressor(pretrained=False)
    if not os.path.isfile(REG_CKPT):
        raise FileNotFoundError(f"Regressor weights not found: {REG_CKPT}")

    # 加载检查点
    checkpoint = torch.load(REG_CKPT, map_location=device)

    # 检查检查点结构并加载
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            reg_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从检查点加载模型，epoch={checkpoint.get('epoch', 'N/A')}")
        elif 'state_dict' in checkpoint:
            reg_model.load_state_dict(checkpoint['state_dict'])
            print(f"从检查点加载模型")
        else:
            # 尝试直接加载为模型权重
            try:
                reg_model.load_state_dict(checkpoint)
                print("加载了直接的模型权重")
            except:
                raise RuntimeError("无法识别检查点格式")
    else:
        # 如果检查点不是字典，直接加载
        reg_model.load_state_dict(checkpoint)
        print("加载了直接的模型权重")

    reg_model.to(device).eval()

    # 3. 定义预处理：RGB+masked_depth → 4ch → [1×4×H×W]
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((TRANSFORM_SIZE, TRANSFORM_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
                    std=[0.229, 0.224, 0.225, 0.25])
    ])

    results = []
    processed_count = 0
    missing_cache_count = 0

    # 用于存储距离信息，用于误差分析
    all_distance_info = []

    # 4. 递归遍历测试文件夹
    print("开始处理测试图像...")
    for dirpath, _, files in os.walk(TEST_IMAGES):
        rel_dir = os.path.relpath(dirpath, TEST_IMAGES)
        for fname in files:
            if not is_image_file(fname):
                continue

            img_path = os.path.join(dirpath, fname)
            image_name = os.path.splitext(fname)[0]
            # 若在子文件夹里，需要在缓存和输出 CSV 中保留子路径
            rel_path = os.path.join(rel_dir, fname) if rel_dir != '.' else fname

            print(f"处理: {rel_path}")

            # 5. 检查缓存文件是否存在
            cache_available, depth_cache_file, inner_mask_file, outer_mask_file = check_cache_files(
                TEST_CACHE, TEST_MASKS, image_name, rel_dir
            )

            if not cache_available:
                print(f"  ⚠ 缓存文件不完整，跳过: {rel_path}")
                missing_cache_count += 1
                continue

            try:
                # 6. 读取原始图像获取尺寸信息
                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"  ⚠ 无法读取图像: {img_path}")
                    continue

                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

                # 7. 从缓存加载深度图和掩码
                depth, inner_mask_uv8, outer_mask_uv8 = load_cached_data(
                    depth_cache_file, inner_mask_file, outer_mask_file, (h, w)
                )

                # 8. 转为布尔掩码
                inner_mask = inner_mask_uv8.astype(bool)
                outer_mask = outer_mask_uv8.astype(bool)

                # 9. 计算平均深度
                inner_avg = safe_compute_avg_depth(depth, inner_mask)
                outer_avg = safe_compute_avg_depth(depth, outer_mask) if outer_mask.any() else 0.0

                # 10. 构造 "圆内深度减去外圈平均" 的图 depth_inner
                depth_inner = (depth - outer_avg) * inner_mask

                # 11. 构造 4 通道输入：RGB + depth_inner
                img4 = np.zeros((h, w, 4), dtype=np.float32)
                img4[:, :, 0] = rgb[:, :, 0].astype(np.float32)
                img4[:, :, 1] = rgb[:, :, 1].astype(np.float32)
                img4[:, :, 2] = rgb[:, :, 2].astype(np.float32)
                img4[:, :, 3] = depth_inner.astype(np.float32)

                # 12. 预处理到 Tensor [1×4×H'×W']
                x4 = transform(img4).unsqueeze(0).to(device)

                # 13. 将 "(inner_avg - outer_avg)" 作为 avgd
                final_avg = inner_avg - outer_avg
                avgd_t = torch.tensor([[final_avg]], dtype=torch.float32, device=device)

                # 14. 前向推理
                with torch.no_grad():
                    pred_distance = reg_model(x4, avgd_t).item()

                # 15. 获取真实标签（如果存在）
                true_distance = None
                if has_labels:
                    # 尝试使用完整文件名查找标签
                    true_distance = label_dict.get(fname)
                    if true_distance is None:
                        # 尝试使用不带扩展名的文件名
                        true_distance = label_dict.get(image_name)

                # 16. 记录结果
                result = {
                    'image': fname,
                    'relative_path': rel_path.replace('\\', '/'),
                    'predicted_distance': pred_distance,
                    'final_avg_depth': final_avg,
                    'inner_avg_depth': inner_avg,
                    'outer_avg_depth': outer_avg
                }

                # 添加真实标签和误差（如果存在）
                if true_distance is not None:
                    result['true_distance'] = true_distance
                    result['distance_error'] = pred_distance - true_distance

                    # 收集距离信息用于误差分析
                    distance_info = {
                        'true_distance': true_distance,
                        'pred_distance': pred_distance,
                        'error': pred_distance - true_distance
                    }
                    all_distance_info.append(distance_info)

                results.append(result)
                processed_count += 1

                # 输出预测信息
                if true_distance is not None:
                    print(
                        f"  ✅ 预测完成: 预测距离={pred_distance:.4f}, 真实距离={true_distance:.4f}, 误差={pred_distance - true_distance:.4f}")
                else:
                    print(f"  ✅ 预测完成: 预测距离={pred_distance:.4f}, 深度={final_avg:.2f}")

            except Exception as e:
                print(f"  ❌ 处理失败 {rel_path}: {str(e)}")
                continue

    # 17. 保存预测结果CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\n预测完成!")
        print(f"成功处理: {processed_count} 个图像")
        print(f"缺少缓存: {missing_cache_count} 个图像")
        print(f"预测结果保存到: {OUTPUT_CSV}")

        # 显示预测统计信息
        if len(results) > 0:
            distances = df['predicted_distance']
            depths = df['final_avg_depth']
            print(f"\n预测统计:")
            print(f"  预测距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
            print(f"  预测距离均值: {distances.mean():.4f} ± {distances.std():.4f}")
            print(f"  深度范围: [{depths.min():.2f}, {depths.max():.2f}]")
            print(f"  深度均值: {depths.mean():.2f} ± {depths.std():.2f}")

        # 18. 如果有距离数据，进行误差分析并生成表9
        if all_distance_info:
            print(f"\n开始误差分析并生成表9...")
            full_stats, table9_df = analyze_prediction_errors(all_distance_info, ERROR_STATS_CSV)

            if full_stats:
                print(f"\n误差分析统计:")
                print(f"  预测平均绝对误差(MAE): {full_stats['预测平均绝对误差(MAE)']:.2f} mm")
                print(f"  预测误差标准差: {full_stats['预测误差标准差']:.2f} mm")
                print(
                    f"  95%预测误差区间: [{full_stats['95%预测误差区间'][0]:.2f}, {full_stats['95%预测误差区间'][1]:.2f}] mm")
                print(f"  RMSE: {full_stats['RMSE']:.2f} mm")
                print(f"  相对误差均值: {full_stats['相对误差均值(%)']:.1f} %")

                # 打印表9内容
                print(f"\n表9 螺栓到表面距离预测误差统计（测试集）")
                print("-" * 60)
                print(f"{'统计指标':<30} {'计算公式':<50} {'数值'}")
                print("-" * 60)
                for _, row in table9_df.iterrows():
                    # 根据列名获取数值
                    if '数值 (mm)' in row:
                        value = row['数值 (mm)']
                    elif '数值 (%)' in row:
                        value = row['数值 (%)']
                    else:
                        value = 'N/A'
                    print(f"{row['统计指标']:<30} {row['计算公式']:<50} {value}")
                print("-" * 60)
        else:
            print("\n警告：没有距离标签数据，无法进行误差分析")

        # 输出目录结构
        print(f"\n输出文件:")
        print(f"{os.path.dirname(OUTPUT_CSV)}/")
        print(f"├── {os.path.basename(OUTPUT_CSV)}")
        if all_distance_info:
            print(f"├── {os.path.basename(ERROR_STATS_CSV)}")
            print(f"├── {os.path.basename(ERROR_REPORT_JSON)}")
            print(f"└── {os.path.basename(ERROR_REPORT_TXT)}")

    else:
        print("❌ 没有成功处理任何图像!")


if __name__ == "__main__":
    main()