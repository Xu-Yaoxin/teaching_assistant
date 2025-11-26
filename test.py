import json
import os

# 检查文件是否存在
tree_path = 'tree.json'
if not os.path.exists(tree_path):
    print("❌ tree.json not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
else:
    print(f"✅ tree.json found at: {os.path.abspath(tree_path)}")
    
    # 检查文件内容
    with open(tree_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查字段名称
    def check_fields(node, path=""):
        current_path = f"{path}-{node.get('name', 'NO_NAME')}" if path else node.get('name', 'ROOT')
        
        if 'children' not in node or not node['children']:
            # 这是叶子节点
            if 'history_records' in node:
                count = len(node['history_records'])
                if count > 0:
                    print(f"  ✅ {current_path}: {count} records")
            else:
                print(f"  ⚠️  {current_path}: missing 'history_records' field")
        else:
            for child in node['children']:
                check_fields(child, current_path)
    
    print("\n📊 Checking leaf nodes:")
    check_fields(data)