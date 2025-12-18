import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import cv2
import glob
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2


def is_image_file(path):
    return any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])


def load_benchmark_data(benchmark_file):
    """
    加载基准点数据
    格式: image(带扩展名), x, y, depth_mm
    示例: 1.bmp, 1224, 1024, 141.5
    """
    if not os.path.exists(benchmark_file):
        print(f"警告: 基准点文件 {benchmark_file} 不存在")
        return {}

    try:
        # 首先尝试读取文件并检查列名
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"文件第一行: {first_line}")

        # 尝试多种分隔符
        for sep in ['\t', ',', ' ', ';']:
            try:
                df = pd.read_csv(benchmark_file, sep=sep, engine='python')
                print(f"使用分隔符 '{repr(sep)}' 读取，列名: {list(df.columns)}")

                # 检查是否有必需的列
                if 'image' in df.columns:
                    break
                else:
                    # 尝试查找相似的列名
                    for col in df.columns:
                        if 'image' in col.lower():
                            df = df.rename(columns={col: 'image'})
                            break
            except Exception as e:
                continue

        # 如果仍然没有image列，使用第一列作为image
        if 'image' not in df.columns and len(df.columns) >= 4:
            print(f"重命名列: {df.columns[0]} -> image")
            df = df.rename(columns={df.columns[0]: 'image'})
            df = df.rename(columns={df.columns[1]: 'x'})
            df = df.rename(columns={df.columns[2]: 'y'})
            df = df.rename(columns={df.columns[3]: 'depth_mm'})

        # 去除列名中的空格
        df.columns = df.columns.str.strip()

    except Exception as e:
        print(f"读取基准点文件时出错: {e}")
        return {}

    benchmark_dict = {}

    # 打印数据框的前几行
    print(f"数据框列名: {list(df.columns)}")
    print(f"数据框前3行:\n{df.head(3)}")

    for _, row in df.iterrows():
        # 尝试获取image列，支持不同的大小写和可能的列名
        if 'image' in row:
            image_name = str(row['image'])
        elif 'Image' in row:
            image_name = str(row['Image'])
        else:
            # 尝试使用第一列
            image_name = str(row.iloc[0])

        # 提取基本文件名（不带扩展名）
        base_name = os.path.splitext(image_name)[0]

        if base_name not in benchmark_dict:
            benchmark_dict[base_name] = {
                'points': [],
                'depths_mm': []
            }

        # 尝试获取x, y, depth_mm
        try:
            x = int(float(row['x'])) if 'x' in row else int(float(row.iloc[1]))
            y = int(float(row['y'])) if 'y' in row else int(float(row.iloc[2]))
            depth_mm = float(row['depth_mm']) if 'depth_mm' in row else float(row.iloc[3])
        except Exception as e:
            print(f"解析行数据出错: {row}")
            continue

        benchmark_dict[base_name]['points'].append([x, y])
        benchmark_dict[base_name]['depths_mm'].append(depth_mm)

    print(f"已加载 {len(benchmark_dict)} 张图像的基准点数据")
    return benchmark_dict


def convert_to_serializable(obj):
    """
    将numpy类型转换为Python原生类型，以便JSON序列化
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def relative_to_absolute_depth(relative_depth, benchmark_depths, benchmark_coords):
    """
    通过基准点将相对深度图转换为绝对深度图(mm)
    使用最小二乘法拟合仿射变换：d_abs = a * d_rel + b
    """
    # 从相对深度图中提取基准点处的深度值
    relative_values = []
    for (x, y) in benchmark_coords:
        if 0 <= x < relative_depth.shape[1] and 0 <= y < relative_depth.shape[0]:
            relative_values.append(relative_depth[y, x])

    if len(relative_values) < 2:
        print("警告：基准点数量不足，无法拟合仿射变换")
        return relative_depth, (1.0, 0.0)

    # 使用最小二乘法拟合线性变换
    relative_values = np.array(relative_values)
    benchmark_depths = np.array(benchmark_depths)

    # 构建矩阵 A * [a, b]^T = b
    A = np.vstack([relative_values, np.ones(len(relative_values))]).T
    params, _, _, _ = np.linalg.lstsq(A, benchmark_depths, rcond=None)

    a, b = float(params[0]), float(params[1])

    # 应用仿射变换到整个深度图
    absolute_depth = a * relative_depth + b

    return absolute_depth, (a, b)


def analyze_depth_errors(absolute_depth, benchmark_coords, benchmark_depths):
    """
    分析深度估计误差，计算完整统计指标
    """
    estimated_depths = []
    absolute_errors = []
    relative_errors = []

    for (x, y), true_depth in zip(benchmark_coords, benchmark_depths):
        if 0 <= x < absolute_depth.shape[1] and 0 <= y < absolute_depth.shape[0]:
            estimated = float(absolute_depth[y, x])
            estimated_depths.append(estimated)

            # 计算绝对误差
            abs_error = abs(estimated - true_depth)
            absolute_errors.append(float(abs_error))

            # 计算相对误差（百分比），避免除零
            if true_depth > 0:
                rel_error = (abs_error / true_depth) * 100
                relative_errors.append(float(rel_error))

    if absolute_errors:
        # 转换为numpy数组便于计算
        abs_errors = np.array(absolute_errors)

        # 1. 平均绝对误差 (MAE)
        mae = float(np.mean(abs_errors))

        # 2. 误差标准差 (Std)
        std_error = float(np.std(abs_errors))

        # 3. 95% 误差上限
        error_95 = float(mae + 1.96 * std_error)

        # 4. RMSE（均方根误差）
        rmse = float(np.sqrt(np.mean(np.square(abs_errors))))

        # 5. 相对误差均值
        mean_relative_error = float(np.mean(relative_errors)) if relative_errors else 0.0

        stats = {
            # 基本统计
            'num_points': int(len(absolute_errors)),
            'mae': mae,
            'std_error': std_error,
            'error_95': error_95,
            'rmse': rmse,

            # 相对误差统计
            'mean_relative_error': mean_relative_error,

            # 极值统计
            'max_absolute_error': float(np.max(abs_errors)) if len(abs_errors) > 0 else 0.0,
            'min_absolute_error': float(np.min(abs_errors)) if len(abs_errors) > 0 else 0.0,

            # 误差值列表（用于进一步分析）
            'absolute_errors': absolute_errors,
            'relative_errors': relative_errors,
        }

        return stats, absolute_errors, relative_errors, estimated_depths

    return None, [], [], []


def generate_error_report(all_individual_stats, output_path):
    """
    生成完整的误差分析报告
    """
    if not all_individual_stats:
        print("警告：没有可用的统计数据进行报告生成")
        return None, None

    # 将numpy类型转换为Python原生类型
    all_individual_stats_serializable = [convert_to_serializable(stats) for stats in all_individual_stats]

    # 创建汇总统计
    df_stats = pd.DataFrame(all_individual_stats_serializable)

    # 计算总体统计 - 确保使用Python原生类型
    summary = {
        '总样本数': int(len(df_stats)),
        '总基准点数': int(df_stats['num_points'].sum()),

        # 绝对误差统计 - 使用float()确保转换为Python float
        '全局平均绝对误差': float(np.mean(df_stats['mae'])),
        '全局误差标准差': float(np.mean(df_stats['std_error'])),
        '全局95%误差上限': float(np.mean(df_stats['mae']) + 1.96 * np.mean(df_stats['std_error'])),
        '全局RMSE': float(np.mean(df_stats['rmse'])),

        # 相对误差统计
        '全局相对误差均值': float(np.mean(df_stats['mean_relative_error'])),
    }

    # 转换为可序列化的Python原生类型
    summary = convert_to_serializable(summary)

    # 保存汇总报告
    report_file = os.path.join(output_path, "深度估计误差分析报告.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 创建详细统计表格
    detailed_stats_df = pd.DataFrame(all_individual_stats_serializable)

    # 保存为CSV
    csv_file = os.path.join(output_path, "深度估计详细统计.csv")
    detailed_stats_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 生成文本报告
    txt_report = os.path.join(output_path, "误差分析报告.txt")
    with open(txt_report, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("深度估计不确定性分析与误差传播报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 总体统计\n")
        f.write("   - 总样本数: {}个\n".format(summary['总样本数']))
        f.write("   - 总基准点数: {}个\n".format(summary['总基准点数']))
        f.write("   - 全局平均绝对误差(MAE): {:.2f}mm\n".format(summary['全局平均绝对误差']))
        f.write("   - 全局误差标准差: {:.2f}mm\n".format(summary['全局误差标准差']))
        f.write("   - 全局95%误差上限: {:.2f}mm\n".format(summary['全局95%误差上限']))
        f.write("   - 全局RMSE: {:.2f}mm\n".format(summary['全局RMSE']))
        f.write("   - 全局相对误差均值: {:.1f}%\n\n".format(summary['全局相对误差均值']))

        f.write("2. 各样本统计指标\n")
        f.write("-" * 80 + "\n")
        for i, stats in enumerate(all_individual_stats_serializable):
            f.write(f"   {i + 1:2d}. {stats['image_name']}:\n")
            f.write(f"       平均绝对误差: {stats['mae']:.2f}mm, ")
            f.write(f"误差标准差: {stats['std_error']:.2f}mm, ")
            f.write(f"95%误差上限: {stats['error_95']:.2f}mm\n")
            f.write(f"       RMSE: {stats['rmse']:.2f}mm, ")
            f.write(f"相对误差: {stats['mean_relative_error']:.1f}%\n")

    # 生成表8格式的统计
    final_stats = pd.DataFrame({
        '统计指标': ['平均绝对误差(MAE)', '误差标准差(Std)', '95%误差上限', 'RMSE', '相对误差均值(%)'],
        '值': [
            f"{summary['全局平均绝对误差']:.2f} mm",
            f"{summary['全局误差标准差']:.2f} mm",
            f"{summary['全局95%误差上限']:.2f} mm",
            f"{summary['全局RMSE']:.2f} mm",
            f"{summary['全局相对误差均值']:.1f} %"
        ],
        '计算公式': [
            r'\(\frac{1}{N} \sum_{i=1}^N |d_i - \hat{d}_i|\)',
            r'\(\sqrt{\frac{1}{N-1} \sum_{i=1}^N (|d_i - \hat{d}_i| - MAE)^2}\)',
            'MAE + 1.96 × Std',
            r'\(\sqrt{\frac{1}{N} \sum_{i=1}^N (d_i - \hat{d}_i)^2}\)',
            r'\(\frac{1}{N} \sum_{i=1}^N \frac{|d_i - \hat{d}_i|}{d_i} \times 100\%\)'
        ]
    })

    table8_file = os.path.join(output_path, "表8_深度估计误差统计分析.csv")
    final_stats.to_csv(table8_file, index=False, encoding='utf-8-sig')

    return summary, detailed_stats_df


if __name__ == '__main__':
    # 硬编码路径
    input_path = r"C:/Users/wjc/Desktop/xue/229lib/shijue/shen/depth_anything_screw_detection/depth_anything_screw_detection/data/test/time"
    benchmark_file = r"C:/Users/wjc/Desktop/xue/229lib/shijue/shen/depth_anything_screw_detection/depth_anything_screw_detection/data/test/Dwucha.csv"  # 基准点数据文件

    # 主输出目录
    main_output_path = r"C:\Users\wjc\Desktop\xue\229lib\shijue\shen\depth_anything_screw_detection\depth_anything_screw_detection\data\test\Djieguo"

    # 子目录
    output_npy_path = os.path.join(main_output_path, "相对深度图1")
    output_absolute_npy_path = os.path.join(main_output_path, "绝对深度图1")
    output_png_path = os.path.join(main_output_path, "可视化结果1")
    output_error_stats_path = os.path.join(main_output_path, "error_stats1")

    # 设置其他参数
    input_size = 518
    encoder_type = 'vitl'
    save_visualization = False  # 启用PNG保存
    grayscale_visualization = False
    save_absolute_depth = False  # 保存绝对深度图

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 加载模型
    depth_anything = DepthAnythingV2(**model_configs[encoder_type])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 获取图像文件列表
    filenames = sorted([f for f in glob.glob(os.path.join(input_path, '**/*'), recursive=True) if is_image_file(f)])

    # 加载基准点数据
    benchmark_data = load_benchmark_data(benchmark_file)

    # 创建输出目录
    os.makedirs(output_npy_path, exist_ok=False)
    os.makedirs(output_absolute_npy_path, exist_ok=False)
    os.makedirs(output_png_path, exist_ok=False)
    os.makedirs(output_error_stats_path, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    all_individual_stats = []

    for k, filename in enumerate(filenames):
        print(f'处理 {k + 1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"警告: 无法加载 {filename}")
            continue

        # 推理深度图
        depth = depth_anything.infer_image(raw_image, input_size)  # float32深度，单位为米

        # 生成输出文件名（保持原文件名）
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # 保存相对深度图 (.npy)
        relative_npy_path = os.path.join(output_npy_path, base_name + '.npy')
        np.save(relative_npy_path, depth)

        # 处理基准点数据
        if base_name in benchmark_data:
            bench_points = benchmark_data[base_name]
            bench_coords = bench_points['points']
            bench_depths = bench_points['depths_mm']

            print(f"  找到基准点: {len(bench_depths)} 个点")

            # 转换为绝对深度
            absolute_depth, transform_params = relative_to_absolute_depth(
                depth, bench_depths, bench_coords
            )

            # 分析误差
            stats, errors, relative_errors, estimated_depths = analyze_depth_errors(
                absolute_depth, bench_coords, bench_depths
            )

            if stats:
                # 确保转换数据类型
                stats['image_name'] = base_name
                stats['transform_params'] = {
                    'a': float(transform_params[0]),
                    'b': float(transform_params[1])
                }

                all_individual_stats.append(stats)

                # 保存每个图像的详细统计
                stats_file = os.path.join(output_error_stats_path, f"{base_name}_详细统计.json")
                # 转换stats为可序列化
                stats_serializable = convert_to_serializable(stats)
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_serializable, f, indent=2, ensure_ascii=False)

                print(f"  平均绝对误差: {stats['mae']:.2f}mm, 相对误差: {stats['mean_relative_error']:.1f}%")

            # 保存绝对深度图
            if save_absolute_depth:
                absolute_npy_path = os.path.join(output_absolute_npy_path, base_name + '_绝对.npy')
                np.save(absolute_npy_path, absolute_depth)

        # 保存可视化PNG
        if save_visualization:
            vis_depth = depth.copy()
            vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min()) * 255.0
            vis_depth = vis_depth.astype(np.uint8)

            if grayscale_visualization:
                vis_depth = np.repeat(vis_depth[..., np.newaxis], 3, axis=-1)
            else:
                vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, vis_depth])

            png_output_path = os.path.join(output_png_path, base_name + '.png')
            cv2.imwrite(png_output_path, combined_result)

    print("\n" + "=" * 60)
    print("深度估计处理完成")
    print("=" * 60)

    # 生成误差分析报告
    if all_individual_stats:
        summary, detailed_stats = generate_error_report(
            all_individual_stats,
            output_error_stats_path
        )

        # 打印关键统计结果
        print("\n深度估计误差统计分析结果:")
        print("-" * 40)
        print(f"统计指标         值")
        print("-" * 40)
        print(f"平均绝对误差(MAE):  {summary['全局平均绝对误差']:.2f} mm")
        print(f"误差标准差(Std):    {summary['全局误差标准差']:.2f} mm")
        print(f"95%误差上限:        {summary['全局95%误差上限']:.2f} mm")
        print(f"RMSE:               {summary['全局RMSE']:.2f} mm")
        print(f"相对误差均值:       {summary['全局相对误差均值']:.1f} %")
        print("-" * 40)

        print(f"\n详细统计结果已保存到: {main_output_path}")

        # 输出目录结构
        print("\n输出目录结构:")
        print(f"{main_output_path}/")
        print(f"├── 相对深度图/")
        print(f"├── 绝对深度图/")
        print(f"├── 可视化结果/")
        print(f"└── error_stats/")
        print(f"    ├── 各图像详细统计.json")
        print(f"    ├── 深度估计详细统计.csv")
        print(f"    ├── 误差分析报告.txt")
        print(f"    └── 表8_深度估计误差统计分析.csv")
    else:
        print("警告：没有找到任何基准点数据，无法进行误差分析")