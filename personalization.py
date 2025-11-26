import json
import requests
import time
from typing import Optional
import math

import os

def load_tree_json() -> Optional[dict]:
    """
    Load and parse the tree.json file, validate the root node's validity.
    Returns: parsed tree structure dictionary, returns None if failed.
    """
    # ✅ 修改:向上一级目录查找 tree.json
    current_dir = os.path.dirname(__file__)
    # 如果当前在 ui 子目录,向上查找
    if os.path.basename(current_dir) == 'ui':
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, "tree.json")
    else:
        file_path = os.path.join(current_dir, "tree.json")
    
    print(f"📂 Trying to load tree.json from: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        
        if tree_data.get("name") != "Mathematics":
            print("Error: The root node of tree.json must be 'Mathematics'")
            return None
        
        def validate_tree(node: dict, level: int) -> bool:
            if level in [2, 3] and "children" not in node:
                print(f"Error: Level {level} node {node.get('name')} is missing required 'children' attribute")
                return False
            if level == 4 and "history_records" not in node:
                print(f"Error: Level 4 node {node.get('name')} is missing required 'history_records' attribute")
                return False
            for child in node.get("children", []):
                if not validate_tree(child, level + 1):
                    return False
            return True

        if not validate_tree(tree_data, level=1):
            return None

        return tree_data

    except FileNotFoundError:
        print(f"Error: tree.json file not found (path: {file_path})")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid format of tree.json file (non-standard JSON)")
        return None
    except Exception as e:
        print(f"Failed to load tree.json: {str(e)}")
        return None

def call_llm(question: str, current_selected: str, options_with_number: str) -> int:
    """
    General model call function, returns the selected option number (0 represents no match).
    :param question: User's original question.
    :param current_selected: Currently selected path (for context).
    :param options_with_number: List of options with numbers (separated by new lines).
    :return: Selected option number (int), returns 0 on failure.
    """
    prompt = f"""【Task】Based on the user's mathematics question, select the most relevant category number.
【User's Question】: {question}
【Current Selected Path】: {current_selected}
【Available Categories (Number + Name)】:
{options_with_number}
【Selection Rules】:
1. Only select the most relevant category based on the core knowledge points of the question, no extra explanation needed;
2. If no category matches, output 0;
3. Only return pure numbers (e.g., 1, 2, 0), do not include any text, punctuation, or formatting symbols.
【Output】:"""
    MODEL_NAME = "qwen3:8b"
    payload_template = {
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "think": False
    }
    payload = payload_template.copy()
    payload["prompt"] = prompt
    OLLAMA_BASE_URL = "http://localhost:11434"
    url = f"{OLLAMA_BASE_URL}/api/generate"

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        selected_str = result.get("response", "").strip()

        return int(selected_str) if selected_str.lstrip("-").isdigit() else 0

    except ValueError:
        print(f"Warning: The model returned a non-numeric result ({selected_str}), treated as no match.")
        return 0
    except requests.exceptions.RequestException as e:
        print(f"Error: Model API call failed: {str(e)}")
        return 0
    except Exception as e:
        print(f"Error: Model call exception: {str(e)}")
        return 0

def generate_numbered_options(nodes: list[dict]) -> tuple[list[str], dict[int, dict]]:
    """
    Generate a list of numbered options and a number-to-node mapping for a list of nodes.
    :param nodes: List of nodes (children of each level).
    :return: (List of numbered options, Mapping of numbers to nodes)
    """
    options = []
    node_map = {}
    for idx, node in enumerate(nodes, start=1):
        node_name = node.get("name", "").strip()
        if node_name:  # Filter out empty name nodes
            options.append(f"{idx}. {node_name}")
            node_map[idx] = node
    return options, node_map

def find_knowledge_point(question: str) -> str:
    """
    Core function: Input user's question, match knowledge points layer by layer, return the full path or fallback result.
    :param question: User's mathematics question.
    :return: Complete knowledge point path (e.g., "Mathematics-Algebra-Elementary Algebra-Solving and Application of Linear Equations") or 0 + reason.
    """
    # 1. Load tree structure
    tree_data = load_tree_json()
    if not tree_data:
        return "0 (Tree structure load failed, unable to match knowledge point)"

    domain_nodes = tree_data.get("children", [])
    if not domain_nodes:
        return "0 (No available mathematics domain nodes)"

    # Generate domain options and mapping
    domain_options, domain_map = generate_numbered_options(domain_nodes)
    if not domain_options:
        return "0 (No valid domain options)"

    # Call model to select domain
    selected_domain_num = call_llm(
        question=question,
        current_selected="Mathematics (root node)",
        options_with_number="\n".join(domain_options)
    )
    if selected_domain_num == 0 or selected_domain_num not in domain_map:
        return "0 (No matching mathematics domain found)"

    # Get selected domain information
    selected_domain = domain_map[selected_domain_num]
    domain_name = selected_domain.get("name", "")
    current_path = f"Mathematics-{domain_name}"
    print(current_path)

    # -------------------------- Step 2: Match 3rd level subdomain nodes --------------------------
    subdomain_nodes = selected_domain.get("children", [])
    if not subdomain_nodes:
        return f"0 ({current_path} has no corresponding subdomain)"

    # Generate subdomain options and mapping
    subdomain_options, subdomain_map = generate_numbered_options(subdomain_nodes)
    if not subdomain_options:
        return f"0 ({current_path} has no valid subdomain options)"

    # Call model to select subdomain
    selected_subdomain_num = call_llm(
        question=question,
        current_selected=current_path,
        options_with_number="\n".join(subdomain_options)
    )
    if selected_subdomain_num == 0 or selected_subdomain_num not in subdomain_map:
        return f"0 ({current_path} no matching subdomain found)"

    # Get selected subdomain information
    selected_subdomain = subdomain_map[selected_subdomain_num]
    subdomain_name = selected_subdomain.get("name", "")
    current_path = f"{current_path}-{subdomain_name}"

    # -------------------------- Step 3: Match 4th level knowledge point nodes --------------------------
    knowledge_nodes = selected_subdomain.get("children", [])
    if not knowledge_nodes:
        return f"0 ({current_path} has no corresponding knowledge point)"

    # Generate knowledge point options and mapping
    knowledge_options, knowledge_map = generate_numbered_options(knowledge_nodes)
    if not knowledge_options:
        return f"0 ({current_path} has no valid knowledge point options)"

    # Call model to select knowledge point
    selected_knowledge_num = call_llm(
        question=question,
        current_selected=current_path,
        options_with_number="\n".join(knowledge_options)
    )
    if selected_knowledge_num == 0 or selected_knowledge_num not in knowledge_map:
        return f"0 ({current_path} no matching knowledge point found)"

    # -------------------------- Final processing: Update history and return result --------------------------
    selected_knowledge = knowledge_map[selected_knowledge_num]
    knowledge_name = selected_knowledge.get("name", "")
    full_path = f"{current_path}-{knowledge_name}"
    print(full_path)

    # ✅ 修复: 使用 history_records 而不是 history
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if "history_records" not in selected_knowledge:
        selected_knowledge["history_records"] = []
    selected_knowledge["history_records"].append(timestamp)
    
    # Persist updated tree structure
    try:
        current_dir = os.path.dirname(__file__)
        if os.path.basename(current_dir) == 'ui':
            parent_dir = os.path.dirname(current_dir)
            tree_path = os.path.join(parent_dir, "tree.json")
        else:
            tree_path = os.path.join(current_dir, "tree.json")
            
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully updated history records to: {tree_path}")
    except Exception as e:
        print(f"Warning: Failed to update history: {str(e)}")

    return full_path


def calculate_time_weighted_sum(history_records: list[str]) -> float:
    """
    Calculate the time-weighted sum W using an exponential decay model.
    :param history_records: List of history records (timestamp strings) for the 4th level node.
    :return: Time-weighted sum W (rounded to 4 decimal places).
    """
    k = 0.1  # Decay factor
    current_time = time.localtime()  # Current system time
    total_weight = 0.0
    time_format = "%Y-%m-%d %H:%M:%S"  # Consistent with history record format

    for ts_str in history_records:
        try:
            # Parse timestamp and calculate time difference (in seconds)
            history_time = time.strptime(ts_str, time_format)
            time_diff_sec = time.mktime(current_time) - time.mktime(history_time)
            delta_t = time_diff_sec / (24 * 3600)  # Convert to days

            # Apply time filtering and weight calculation
            if delta_t > 90:
                weight = 0.0
            else:
                weight = math.exp(-k * delta_t)

            total_weight += weight
        except ValueError:
            print(f"Warning: Invalid timestamp format ({ts_str}), skipping this record")
            continue

    return round(total_weight, 4)


def get_attention_level(history_records: list[str]) -> int:
    """
    Return the attention level (0/1/2) based on the time-weighted sum.
    :param history_records: List of history records (timestamp strings) for the 4th level node.
    :return: Attention level (2 = highest attention, 1 = medium attention, 0 = low attention).
    """
    weighted_sum = calculate_time_weighted_sum(history_records)

    # Level thresholds: Nodes with repeated questions in the short term accumulate weight to a higher level.
    if weighted_sum >= 5.0:
        return 2
    elif 2.0 <= weighted_sum < 5.0:
        return 1
    else:
        return 0


def get_question_attention_level(question: str) -> tuple[str, int]:
    """
    Input the user's question, return the matched knowledge point path and corresponding attention level.
    :param question: User's mathematics question.
    :return: (Knowledge point path string, Attention level 0/1/2).
    """
    # Step 1: Get knowledge point match result
    knowledge_path = find_knowledge_point(question)
    if knowledge_path.startswith("0"):
        return knowledge_path, 0  # No knowledge point matched, default level 0

    # Step 2: Load tree structure and find the target node
    tree_data = load_tree_json()
    if not tree_data:
        return knowledge_path, 0

    # Split the path into parts (format: Mathematics-Domain-Subdomain-Knowledge Point)
    path_parts = knowledge_path.split("-")
    if len(path_parts) != 4:
        return knowledge_path, 0  # Path format is abnormal

    domain_name, subdomain_name, knowledge_name = path_parts[1], path_parts[2], path_parts[3]
    target_node = None

    # Traverse each level to find the target knowledge point node
    for domain_node in tree_data.get("children", []):
        if domain_node.get("name") == domain_name:
            for subdomain_node in domain_node.get("children", []):
                if subdomain_node.get("name") == subdomain_name:
                    for knowledge_node in subdomain_node.get("children", []):
                        if knowledge_node.get("name") == knowledge_name:
                            target_node = knowledge_node
                            break
                    break
            break

    # ✅ 修复: 使用 history_records 而不是 history
    if not target_node or "history_records" not in target_node:
        return knowledge_path, 0
    return knowledge_path, get_attention_level(target_node["history_records"])


if __name__ == "__main__":
    test_question = "Evaluate the definite integral for given function"
    # Get path and attention level
    path, level = get_question_attention_level(test_question)
    print(f"Attention level {level}")
