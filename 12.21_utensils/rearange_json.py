import json
import os
import datetime
import time

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format


def rearrange_json(input_path, output_path):
    """
    读取旧JSON，合并所有数据，根据新阈值重新分组保存
    """
    print(f"📂 Loading data from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        old_data = json.load(f)

    # 1. 获取旧的 Metadata (为了保留 weights 信息)
    old_meta = old_data.get("metadata", {})
    weights = old_meta.get("weights", {})
    
    # 2. 【关键步骤】把所有列表合并成一个大 List
    # 注意：我们假设之前的 JSON 里保留了 discarded。
    # 如果之前 discarded 是空的，那我们也只能在现有的 high/medium 里重排。
    all_items = []
    all_items.extend(old_data.get("high_confidence_group", []))
    all_items.extend(old_data.get("medium_confidence_group", []))
    all_items.extend(old_data.get("discarded", []))
    # all_items.extend(old_data.get("group0", []))
    # all_items.extend(old_data.get("group1", []))
    # all_items.extend(old_data.get("group2", []))
    
    total_count = len(all_items)
    print(f"📊 Total items loaded: {total_count}")

    t = [0.55, 0.45, 0.38]

    # 3. 初始化新的容器
    new_output = {
        "metadata": {
            "total_queries": total_count,
            "thresholds" : t,
            "weights": weights, # 继承之前的权重设置
            "size_of_each_group": [0, 0, 0, 0]
        },
        "group0": [],  
        "group1": [], 
        "group2": [],
        "discarded": []        
    }

    # 4. 【重筛】一个个摘出来
    seen_things = set()
    seen_nsd = set()
    for item in all_items:
        score = item["score_final"]
        if item["things_id"] in seen_things:
            continue
        if item["nsd_id"] in seen_nsd:
            continue
        if score >= t[2]: 
            seen_things.add(item["things_id"])
            seen_nsd.add(item["nsd_id"])
        if score >= t[0]:
            new_output["group0"].append(item)
            new_output["metadata"]["size_of_each_group"][0] += 1
        elif score >= t[1]:
            new_output["group1"].append(item)
            new_output["metadata"]["size_of_each_group"][1] += 1
        elif score >= t[2]:
            new_output["group2"].append(item)
            new_output["metadata"]["size_of_each_group"][2] += 1
    print("seen nsd : ", len(seen_nsd))
    print("seen things : ", len(seen_things))
    allow_duplicate = 1
    if allow_duplicate:
        for item in all_items:
            score = item["score_final"]
            if item["things_id"] in seen_things:
                continue
            if score >= t[2]: 
                seen_things.add(item["things_id"])
                seen_nsd.add(item["nsd_id"])
            if score >= t[0]:
                new_output["group0"].append(item)
                new_output["metadata"]["size_of_each_group"][0] += 1
            elif score >= t[1]:
                new_output["group1"].append(item)
                new_output["metadata"]["size_of_each_group"][1] += 1
            elif score >= t[2]:
                new_output["group2"].append(item)
                new_output["metadata"]["size_of_each_group"][2] += 1
    print("After allowing duplicate:")
    print("seen nsd : ", len(seen_nsd))
    print("seen things : ", len(seen_things))
    # 5. 保存结果
    with open(output_path, 'w') as f:
        json.dump(new_output, f, indent=4)

    print(f"✅ Done! Saved to {output_path}")
    print("count : ", new_output["metadata"]["size_of_each_group"])

if __name__ == "__main__":
    # 输入文件
    input_file = "retrieval_a2_t52.json"
    
    # 输出文件 
    output_file = f"retrieval_rearranged_{get_current_time_info()}.json"
    
    rearrange_json(input_file, output_file)