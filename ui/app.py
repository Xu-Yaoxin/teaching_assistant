from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import json
import time
import sqlite3
import datetime
import re
import os
import pytz
import requests
import shutil
import PyPDF2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 数据库文件路径
DB_PATH = 'chat_app.db'

# 文件上传配置
UPLOAD_FOLDER = 'file'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'mp4', 'avi', 'mov'}

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制文件大小为100MB

# 【新增】会话文件夹映射字典（conversation_id -> folder_path）
conversation_folders = {}
conversation_current_batch = {}
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_china_time():
    """获取中国时间（东八区）"""
    tz = pytz.timezone('Asia/Shanghai')
    return datetime.datetime.now(tz)

# 在 init_db() 函数中添加收藏字段
def init_db():
    """初始化数据库和必要的表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建对话表管理表（添加is_pinned、last_message_time和is_favorited字段）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT UNIQUE NOT NULL,
        title TEXT,
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_pinned INTEGER DEFAULT 0,
        last_message_time TIMESTAMP,
        is_favorited INTEGER DEFAULT 0
    )
    ''')
    
    # 为已存在的表添加新字段（如果不存在）
    try:
        cursor.execute('ALTER TABLE conversation_metadata ADD COLUMN is_pinned INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE conversation_metadata ADD COLUMN last_message_time TIMESTAMP')
    except sqlite3.OperationalError:
        pass
    
    # 【新增】添加收藏字段
    try:
        cursor.execute('ALTER TABLE conversation_metadata ADD COLUMN is_favorited INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()

def get_db_connection():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH)

def execute_sql(sql_code, params=None, fetch=False):
    """
    执行SQL语句的通用函数
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(sql_code, params)
        else:
            cursor.execute(sql_code)
        
        if fetch:
            result = cursor.fetchall()
            # 转换为字典列表
            columns = [col[0] for col in cursor.description] if cursor.description else []
            result = [dict(zip(columns, row)) for row in result]
        else:
            conn.commit()
            result = None
            
        cursor.close()
        return result
    except Exception as err:
        conn.rollback()
        raise err
    finally:
        conn.close()

def create_conversation_table(table_name):
    """
    创建对话表 - 只有sentence一列
    """
    # 确保表名只包含字母、数字和下划线
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
    
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence TEXT NOT NULL
    )
    """
    execute_sql(sql)
    
    # 插入初始欢迎消息
    welcome_message = "你好！我是智能助手，有什么我可以帮你的吗？我可以回答各种问题、提供建议、解释概念，或者只是陪你聊天。请随时向我提问！"
    china_time = get_china_time().strftime('%Y-%m-%d %H:%M:%S')
    welcome_with_time = f"{welcome_message}<<<TIME>>>{china_time}"
    insert_sql = f'INSERT INTO "{table_name}" (sentence) VALUES (?)'
    execute_sql(insert_sql, (welcome_with_time,))
    
    # 记录对话表信息（使用中国时间，初始化last_message_time）
    execute_sql(
        'INSERT OR IGNORE INTO conversation_metadata (table_name, created_time, last_message_time, is_pinned) VALUES (?, ?, ?, 0)',
        (table_name, china_time, china_time)
    )
    
    return table_name

def check_table_has_only_welcome(table_name):
    """
    检查表是否只有欢迎消息
    """
    try:
        # 检查记录数量
        count_sql = f'SELECT COUNT(*) as count FROM "{table_name}"'
        result = execute_sql(count_sql, fetch=True)
        
        if result and result[0]['count'] == 1:
            # 检查内容是否是欢迎消息
            content_sql = f'SELECT sentence FROM "{table_name}" WHERE id = 1'
            content_result = execute_sql(content_sql, fetch=True)
            
            welcome_message = "你好！我是智能助手，有什么我可以帮你的吗？我可以回答各种问题、提供建议、解释概念，或者只是陪你聊天。请随时向我提问！"
            if content_result:
                # 移除时间戳后比较
                sentence = content_result[0]['sentence']
                if '<<<TIME>>>' in sentence:
                    sentence = sentence.split('<<<TIME>>>')[0]
                if sentence == welcome_message:
                    return True
        return False
    except:
        return False

def get_all_conversation_tables():
    """
    获取所有的对话表（置顶的排在前面，其他按最后消息时间倒序）
    """
    try:
        # 获取所有对话表信息，置顶的排在前面，然后按最后消息时间倒序
        sql = """
        SELECT table_name, title, created_time, is_pinned, last_message_time
        FROM conversation_metadata 
        ORDER BY is_pinned DESC, 
                 COALESCE(last_message_time, created_time) DESC
        """
        result = execute_sql(sql, fetch=True)
        return result if result else []
    except Exception as e:
        print(f"获取对话表失败: {e}")
        return []

def get_conversation_messages(table_name):
    """
    获取对话表中的所有消息
    """
    try:
        sql = f'SELECT id, sentence FROM "{table_name}" ORDER BY id'
        result = execute_sql(sql, fetch=True)
        return result if result else []
    except Exception as e:
        print(f"获取对话消息失败: {e}")
        return []

def delete_conversation_table(table_name):
    """
    删除对话表
    """
    try:
        # 删除表
        sql = f'DROP TABLE IF EXISTS "{table_name}"'
        execute_sql(sql)
        
        # 删除元数据记录
        execute_sql(
            'DELETE FROM conversation_metadata WHERE table_name = ?',
            (table_name,)
        )
        
        return True
    except Exception as e:
        print(f"删除对话表失败: {e}")
        return False

def generate_conversation_title(first_question, avoid_duplicates=True):
    """
    生成对话标题（基于第一个问题）
    避免重复标题
    """
    try:
        # 移除时间戳
        if '<<<TIME>>>' in first_question:
            first_question = first_question.split('<<<TIME>>>')[0]
        
        # 简化标题生成逻辑
        base_title = first_question[:15] + "..." if len(first_question) > 15 else first_question
        base_title = base_title.replace("\n", " ").strip()
        
        if not avoid_duplicates:
            return base_title
        
        # 检查标题是否重复，如果重复则添加数字后缀
        title = base_title
        counter = 1
        
        while True:
            check_sql = 'SELECT COUNT(*) as count FROM conversation_metadata WHERE title = ?'
            result = execute_sql(check_sql, (title,), fetch=True)
            
            if result and result[0]['count'] > 0:
                # 标题重复，添加数字后缀
                title = f"{base_title} ({counter})"
                counter += 1
            else:
                # 标题不重复，使用此标题
                break
        
        return title
        
    except Exception as e:
        print(f"生成标题失败: {e}")
        return "新对话"

# 初始化数据库
init_db()

@app.route('/')
def index():
    """主页面路由"""
    return render_template('index.html')

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """新建对话路由"""
    try:
        tables = get_all_conversation_tables()
        empty_tables = [table['table_name'] for table in tables if check_table_has_only_welcome(table['table_name'])]
        
        if len(empty_tables) > 1:
            for table_name in empty_tables[1:]:
                delete_conversation_table(table_name)
            empty_tables = [empty_tables[0]]
        
        if empty_tables:
            conversation_id = empty_tables[0]
            truncate_sql = f'DELETE FROM "{conversation_id}" WHERE id > 1'
            execute_sql(truncate_sql)
            reset_sql = f'DELETE FROM sqlite_sequence WHERE name = "{conversation_id}"'
            execute_sql(reset_sql)
            
            # 【新增】重置批次号为1
            conversation_current_batch[conversation_id] = 1
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id
            })
        else:
            timestamp = get_china_time().strftime("%Y%m%d_%H%M%S")
            table_name = f"conversation_{timestamp}"
            created_table = create_conversation_table(table_name)
            
            # 【新增】初始化批次号为1
            conversation_current_batch[created_table] = 1
            
            return jsonify({
                'success': True,
                'conversation_id': created_table
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'创建对话时出错: {str(e)}'
        })
@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    """获取所有对话列表（只显示有实际对话的记录）"""
    try:
        tables = get_all_conversation_tables()
        conversations = []
        
        for table in tables:
            messages = get_conversation_messages(table['table_name'])
            message_count = len(messages)
            
            # 改为消息数量大于等于2（包含欢迎消息+用户第一条消息）
            if message_count >= 2:
                # 优先使用保存的标题
                if table.get('title'):
                    title = table['title']
                else:
                    # 如果没有保存的标题，生成一个
                    first_question = messages[1]['sentence'] if len(messages) > 1 else "新对话"
                    title = generate_conversation_title(first_question)
                
                # 获取最后一条消息的时间（从last_message_time字段）
                last_time = table.get('last_message_time', table['created_time'])
                if isinstance(last_time, str):
                    try:
                        dt = datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
                        formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_date = last_time[:16] if len(last_time) >= 16 else last_time
                else:
                    formatted_date = last_time.strftime('%Y-%m-%d %H:%M') if hasattr(last_time, 'strftime') else str(last_time)
                
                conversations.append({
                    'id': table['table_name'],
                    'title': title,
                    'date': formatted_date,
                    'message_count': message_count - 1,  # 减去欢迎消息
                    'is_pinned': table.get('is_pinned', 0)
                })
        
        return jsonify({
            'success': True,
            'conversations': conversations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取对话列表失败: {str(e)}'
        })



@app.route('/upload_file', methods=['POST'])
def upload_file():
    """上传文件接口 - 支持批次管理"""
    try:
        print("=" * 50)
        print("📤 收到文件上传请求")
        
        if 'file' not in request.files:
            print("❌ 错误: 没有选择文件")
            return jsonify({'success': False, 'message': '没有选择文件'}), 400  # 添加状态码
        
        file = request.files['file']
        conversation_id = request.form.get('conversation_id')
        
        print(f"📁 文件名: {file.filename}")
        print(f"💬 对话ID: {conversation_id}")
        
        if not conversation_id:
            print("❌ 错误: 缺少对话ID")
            return jsonify({'success': False, 'message': '缺少对话ID'}), 400
        
        if file.filename == '':
            print("❌ 错误: 文件名为空")
            return jsonify({'success': False, 'message': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            print(f"❌ 错误: 不支持的文件类型")
            return jsonify({'success': False, 'message': f'不支持的文件类型，仅支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # 【新增】获取或创建新批次编号
        if conversation_id not in conversation_current_batch:
            conversation_current_batch[conversation_id] = 1
        else:
            conversation_current_batch[conversation_id] += 1
        
        current_batch = conversation_current_batch[conversation_id]
        
        # 创建批次文件夹
        conversation_folder = os.path.join(app.config['UPLOAD_FOLDER'], conversation_id)
        batch_folder = os.path.join(conversation_folder, f"batch_{current_batch}")
        os.makedirs(batch_folder, exist_ok=True)
        print(f"📂 批次文件夹: {batch_folder} (批次 {current_batch})")
        
        conversation_folders[conversation_id] = batch_folder
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(batch_folder, filename)
        
        file_exists = os.path.exists(filepath)
        
        if file_exists:
            print(f"⚠️ 文件已存在，将覆盖: {filepath}")
            try:
                delete_response = requests.post(
                    'http://localhost:8000/delete-document',
                    json={'path': filepath},
                    timeout=10
                )
                print(f"🗑️ 删除旧向量数据: {delete_response.json()}")
            except Exception as e:
                print(f"⚠️ 删除旧向量数据失败（可忽略）: {e}")
        
        # 保存文件
        file.save(filepath)
        print(f"✅ 文件已保存: {filepath}")
        
        # 【修改】调用后端API时增加超时时间和错误处理
        try:
            print("📚 开始加载文档到知识库...")
            load_response = requests.post(
                'http://localhost:8000/load-documents',
                json={'path': filepath},
                timeout=120  # 增加超时到120秒
            )
            
            if load_response.status_code == 200:
                load_result = load_response.json()
                print(f"✅ 知识库加载结果: {load_result}")
                
                if load_result.get('status') == 'success':
                    action = '重新上传' if file_exists else '上传'
                    print(f"✅ {action}成功并已加载到知识库")
                    print("=" * 50)
                    return jsonify({
                        'success': True,
                        'message': f'文件{action}成功并已加载到知识库（批次 {current_batch}）',
                        'filename': filename,
                        'filepath': filepath,
                        'batch_number': current_batch,
                        'load_info': load_result.get('message', '')
                    }), 200
                else:
                    action = '已重新上传' if file_exists else '已上传'
                    print(f"⚠️ 文件{action}，但加载失败: {load_result.get('message')}")
                    return jsonify({
                        'success': True,
                        'message': f'文件{action}到服务器（批次 {current_batch}）',
                        'filename': filename,
                        'filepath': filepath,
                        'batch_number': current_batch
                    }), 200
            else:
                print(f"⚠️ 知识库API返回错误状态码: {load_response.status_code}")
                return jsonify({
                    'success': True,
                    'message': f'文件已上传到服务器（批次 {current_batch}）',
                    'filename': filename,
                    'filepath': filepath,
                    'batch_number': current_batch
                }), 200
                
        except requests.exceptions.Timeout:
            print("⏰ 知识库加载超时（文件已保存）")
            return jsonify({
                'success': True,
                'message': f'文件已上传，知识库加载中（批次 {current_batch}）',
                'filename': filename,
                'filepath': filepath,
                'batch_number': current_batch
            }), 200
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到知识库服务（文件已保存）")
            return jsonify({
                'success': True,
                'message': f'文件已上传，但知识库服务未响应（批次 {current_batch}）',
                'filename': filename,
                'filepath': filepath,
                'batch_number': current_batch,
                'warning': 'Knowledge base service not available'
            }), 200
        except Exception as e:
            print(f"❌ 加载文档异常: {str(e)}")
            return jsonify({
                'success': True,
                'message': f'文件已上传（批次 {current_batch}）',
                'filename': filename,
                'filepath': filepath,
                'batch_number': current_batch,
                'warning': str(e)
            }), 200
        
    except Exception as e:
        print(f"❌ 文件上传失败: {str(e)}")
        print("=" * 50)
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        return jsonify({
            'success': False, 
            'message': f'文件上传失败: {str(e)}'
        }), 500

# 修改 get_file_content 函数
@app.route('/get_file_content', methods=['POST'])
def get_file_content():
    """获取当前批次中已上传文件的内容"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        filename = data.get('filename')
        
        if not conversation_id or not filename:
            return jsonify({'success': False, 'message': '缺少必要参数', 'content': ''})
        
        # 构建当前批次的文件路径
        filename = secure_filename(filename)
        current_batch = conversation_current_batch.get(conversation_id, 1)
        conversation_folder = os.path.join(UPLOAD_FOLDER, conversation_id)
        batch_folder = os.path.join(conversation_folder, f"batch_{current_batch}")
        filepath = os.path.join(batch_folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': '文件不存在', 'content': ''})
        
        # 读取文件内容（保持原有逻辑）
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        content = ''
        
        if file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif file_ext == 'pdf':
            try:
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                    content = '\n\n'.join(text_parts)
            except Exception as e:
                return jsonify({'success': False, 'message': f'PDF读取失败: {str(e)}', 'content': ''})
        elif file_ext in ['doc', 'docx']:
            try:
                from unstructured.partition.auto import partition
                elements = partition(filepath, language="zh")
                content = '\n'.join([elem.text.strip() for elem in elements if hasattr(elem, 'text') and elem.text.strip()])
            except ImportError:
                return jsonify({'success': False, 'message': 'unstructured库未安装，无法读取Word文件', 'content': ''})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Word读取失败: {str(e)}', 'content': ''})
        elif file_ext in ['ppt', 'pptx']:
            try:
                from unstructured.partition.auto import partition
                elements = partition(filepath, language="zh")
                content = '\n'.join([elem.text.strip() for elem in elements if hasattr(elem, 'text') and elem.text.strip()])
            except ImportError:
                return jsonify({'success': False, 'message': 'unstructured库未安装，无法读取PPT文件', 'content': ''})
            except Exception as e:
                return jsonify({'success': False, 'message': f'PPT读取失败: {str(e)}', 'content': ''})
        else:
            return jsonify({'success': False, 'message': f'不支持的文件类型: {file_ext}', 'content': ''})
        
        if not content.strip():
            return jsonify({'success': False, 'message': '文件内容为空', 'content': ''})
        
        return jsonify({'success': True, 'content': content, 'message': '读取成功'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'读取文件失败: {str(e)}', 'content': ''})




@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    """获取当前批次的已上传文件列表"""
    try:
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'success': False, 'message': '缺少对话ID'})
        
        # 【修改】只读取当前批次的文件夹
        current_batch = conversation_current_batch.get(conversation_id, 1)
        conversation_folder = os.path.join(UPLOAD_FOLDER, conversation_id)
        batch_folder = os.path.join(conversation_folder, f"batch_{current_batch}")
        
        files = []
        if os.path.exists(batch_folder):
            for filename in os.listdir(batch_folder):
                filepath = os.path.join(batch_folder, filename)
                if os.path.isfile(filepath):
                    file_stat = os.stat(filepath)
                    files.append({
                        'name': filename,
                        'size': file_stat.st_size,
                        'upload_time': datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # 按上传时间倒序排列
        files.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files,
            'current_batch': current_batch  # 【新增】返回当前批次号
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取文件列表失败: {str(e)}'})
    
# 修改 delete_file 函数
@app.route('/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    """删除当前批次中的已上传文件"""
    try:
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'success': False, 'message': '缺少对话ID'})
        
        # 安全检查，防止路径穿越攻击
        filename = secure_filename(filename)
        current_batch = conversation_current_batch.get(conversation_id, 1)
        conversation_folder = os.path.join(UPLOAD_FOLDER, conversation_id)
        batch_folder = os.path.join(conversation_folder, f"batch_{current_batch}")
        filepath = os.path.join(batch_folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': '文件不存在'})
        
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'文件 {filename} 已删除'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'删除文件失败: {str(e)}'})

@app.route('/load_conversation/<conversation_id>', methods=['GET'])
def load_conversation(conversation_id):
    """加载特定对话的所有消息"""
    try:
        messages = get_conversation_messages(conversation_id)
        
        # 按照奇偶行分配角色
        formatted_messages = []
        for i, msg in enumerate(messages):
            # 奇数行（索引从0开始，所以id为奇数的行索引是偶数）是机器人
            # 偶数行（索引从0开始，所以id为偶数的行索引是奇数）是用户
            role = 'ai' if i % 2 == 0 else 'user'
            formatted_messages.append({
                'role': role,
                'content': msg['sentence']
            })
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': formatted_messages
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'加载对话失败: {str(e)}'
        })

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """删除特定对话"""
    try:
        # 【新增】先删除对话对应的文件夹
        conversation_folder = os.path.join(UPLOAD_FOLDER, conversation_id)
        if os.path.exists(conversation_folder):
            import shutil
            try:
                shutil.rmtree(conversation_folder)
                print(f"✅ 已删除对话文件夹: {conversation_folder}")
            except Exception as e:
                print(f"⚠️ 删除文件夹失败: {str(e)}")
        
        # 原有的删除对话表逻辑
        success = delete_conversation_table(conversation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '对话已成功删除',
                'deleted_id': conversation_id
            })
        else:
            return jsonify({
                'success': False,
                'message': '删除对话失败'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除对话时出错: {str(e)}'
        })
@app.route('/save_message', methods=['POST'])
def save_message():
    """保存消息到数据库"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        message = data.get('message')
        role = data.get('role', 'ai')
        
        if not conversation_id or not message:
            return jsonify({'success': False, 'message': '缺少必要参数'})
        
        # 确保对话表存在
        try:
            check_sql = f'SELECT 1 FROM "{conversation_id}" LIMIT 1'
            execute_sql(check_sql, fetch=True)
        except:
            # 表不存在，创建新表
            create_conversation_table(conversation_id)
        
        # 获取中国时间并格式化
        china_time = get_china_time().strftime('%Y-%m-%d %H:%M:%S')
        
        # 在消息后面添加时间戳（使用特殊分隔符）
        message_with_time = f"{message}<<<TIME>>>{china_time}"
        
        # 保存消息
        insert_sql = f'INSERT INTO "{conversation_id}" (sentence) VALUES (?)'
        execute_sql(insert_sql, (message_with_time,))
        
        # 【新增】更新last_message_time
        execute_sql(
            'UPDATE conversation_metadata SET last_message_time = ? WHERE table_name = ?',
            (china_time, conversation_id)
        )
        
        # 如果是用户的第一条消息（表中只有欢迎消息），自动生成标题
        if role == 'user':
            count_sql = f'SELECT COUNT(*) as count FROM "{conversation_id}"'
            count_result = execute_sql(count_sql, fetch=True)
            
            # 如果保存后只有2条消息（欢迎消息+这条用户消息），生成标题
            if count_result and count_result[0]['count'] == 2:
                # 检查是否已有标题
                check_title_sql = 'SELECT title FROM conversation_metadata WHERE table_name = ?'
                title_result = execute_sql(check_title_sql, (conversation_id,), fetch=True)
                
                if title_result and not title_result[0]['title']:
                    # 生成不重复的标题
                    new_title = generate_conversation_title(message, avoid_duplicates=True)
                    execute_sql(
                        'UPDATE conversation_metadata SET title = ? WHERE table_name = ?',
                        (new_title, conversation_id)
                    )
        
        return jsonify({'success': True, 'message': '消息保存成功', 'timestamp': china_time})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存消息失败: {str(e)}'})


@app.route('/toggle_pin/<conversation_id>', methods=['PUT'])
def toggle_pin(conversation_id):
    """切换对话的置顶状态"""
    try:
        # 获取当前置顶状态
        check_sql = 'SELECT is_pinned FROM conversation_metadata WHERE table_name = ?'
        result = execute_sql(check_sql, (conversation_id,), fetch=True)
        
        if not result:
            return jsonify({'success': False, 'message': '对话不存在'})
        
        current_pinned = result[0]['is_pinned']
        new_pinned = 0 if current_pinned else 1
        
        # 更新置顶状态
        execute_sql(
            'UPDATE conversation_metadata SET is_pinned = ? WHERE table_name = ?',
            (new_pinned, conversation_id)
        )
        
        return jsonify({
            'success': True,
            'message': '置顶状态已更新',
            'is_pinned': new_pinned
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'更新置顶状态失败: {str(e)}'})

@app.route('/toggle_favorite/<conversation_id>', methods=['PUT'])
def toggle_favorite(conversation_id):
    """切换对话的收藏状态"""
    try:
        # 获取当前收藏状态
        check_sql = 'SELECT is_favorited FROM conversation_metadata WHERE table_name = ?'
        result = execute_sql(check_sql, (conversation_id,), fetch=True)
        
        if not result:
            return jsonify({'success': False, 'message': '对话不存在'})
        
        current_favorited = result[0]['is_favorited']
        new_favorited = 0 if current_favorited else 1
        
        # 更新收藏状态
        execute_sql(
            'UPDATE conversation_metadata SET is_favorited = ? WHERE table_name = ?',
            (new_favorited, conversation_id)
        )
        
        return jsonify({
            'success': True,
            'message': '收藏状态已更新',
            'is_favorited': new_favorited
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'更新收藏状态失败: {str(e)}'})
# 【新增】获取收藏列表的路由
@app.route('/get_favorites', methods=['GET'])
def get_favorites():
    """获取所有收藏的对话列表"""
    try:
        tables = get_all_conversation_tables()
        favorites = []
        
        for table in tables:
            # 只返回收藏的对话
            if table.get('is_favorited', 0) == 1:
                messages = get_conversation_messages(table['table_name'])
                message_count = len(messages)
                
                if message_count >= 2:
                    if table.get('title'):
                        title = table['title']
                    else:
                        first_question = messages[1]['sentence'] if len(messages) > 1 else "新对话"
                        title = generate_conversation_title(first_question)
                    
                    last_time = table.get('last_message_time', table['created_time'])
                    if isinstance(last_time, str):
                        try:
                            dt = datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
                            formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_date = last_time[:16] if len(last_time) >= 16 else last_time
                    else:
                        formatted_date = last_time.strftime('%Y-%m-%d %H:%M') if hasattr(last_time, 'strftime') else str(last_time)
                    
                    favorites.append({
                        'id': table['table_name'],
                        'title': title,
                        'date': formatted_date,
                        'message_count': message_count - 1,
                        'is_pinned': table.get('is_pinned', 0)
                    })
        
        return jsonify({
            'success': True,
            'favorites': favorites
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取收藏列表失败: {str(e)}'
        })


@app.route('/update_conversation_title/<conversation_id>', methods=['PUT'])
def update_conversation_title(conversation_id):
    """更新对话标题"""
    try:
        data = request.json
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({'success': False, 'message': '标题不能为空'})
        
        # 【新增】检查标题是否已存在
        check_sql = 'SELECT COUNT(*) as count FROM conversation_metadata WHERE title = ? AND table_name != ?'
        result = execute_sql(check_sql, (new_title, conversation_id), fetch=True)
        
        if result and result[0]['count'] > 0:
            return jsonify({'success': False, 'message': '标题已存在，请使用其他标题'})
        
        # 更新元数据表中的标题
        execute_sql(
            'UPDATE conversation_metadata SET title = ? WHERE table_name = ?',
            (new_title, conversation_id)
        )
        
        return jsonify({'success': True, 'message': '标题更新成功'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'更新标题失败: {str(e)}'})
from flask import render_template

@app.route('/graphing')
def graphing():
    return render_template('graphing.html')
@app.route('/api/recognize-content', methods=['POST'])

def proxy_recognize_content():
    """代理文件识别请求到FastAPI后端"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件', 'result': ''})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空', 'result': ''})
        
        # 准备转发到FastAPI的文件数据
        files = {'file': (file.filename, file.stream, file.content_type)}
        
        # 转发到FastAPI后端
        response = requests.post(
            'http://localhost:8000/recognize-content',
            files=files,
            timeout=30  # 设置30秒超时
        )
        
        # 直接返回FastAPI的响应
        return jsonify(response.json())
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'message': '无法连接到后端服务，请确保FastAPI服务正在运行',
            'result': ''
        })
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'message': '识别超时，请尝试使用更小的文件',
            'result': ''
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'代理请求失败: {str(e)}',
            'result': ''
        })
@app.route('/report_preference', methods=['GET'])
def report_preference():
    """Get top 5 most frequently asked topics"""
    try:
        # ✅ 修改:向上一级目录查找 tree.json
        current_dir = os.path.dirname(__file__)  # ui 文件夹
        parent_dir = os.path.dirname(current_dir)  # comp4431 文件夹
        tree_path = os.path.join(parent_dir, 'tree.json')
        
        print(f"📂 Loading tree.json from: {tree_path}")
        
        if not os.path.exists(tree_path):
            print("❌ tree.json not found!")
            return jsonify({
                'success': False, 
                'topics': [],
                'error': 'tree.json file not found'
            }), 404
        
        with open(tree_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        print(f"✅ tree.json loaded successfully")
        
        def find_leaf_nodes_with_history(node, path=""):
            """Recursively find all leaf nodes with history"""
            leaf_nodes = []
            current_path = f"{path}-{node['name']}" if path else node['name']
            
            if 'children' not in node or not node['children']:
                history_count = len(node.get('history_records', []))
                if history_count > 0:
                    leaf_nodes.append((current_path, history_count))
                    print(f"  📊 {current_path}: {history_count} records")
            else:
                for child in node['children']:
                    leaf_nodes.extend(find_leaf_nodes_with_history(child, current_path))
            
            return leaf_nodes
        
        all_leaf_nodes = find_leaf_nodes_with_history(tree_data)
        sorted_nodes = sorted(all_leaf_nodes, key=lambda x: (-x[1], x[0]))
        top_five = [node[0] for node in sorted_nodes[:5]]
        
        print(f"🏆 Top 5 topics: {top_five}")
        
        return jsonify({
            'success': True,
            'topics': top_five
        })
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return jsonify({
            'success': False, 
            'topics': [],
            'error': 'tree.json file not found'
        }), 404
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return jsonify({
            'success': False, 'topics': [],
            'error': 'Invalid JSON format'
        }), 500
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return jsonify({
            'success': False, 
            'topics': [],
            'error': str(e)
        }), 500
    
@app.route('/api/ask-stream', methods=['POST'])
def proxy_ask_stream():
    """代理请求到FastAPI后端（添加知识点匹配）"""
    try:
        data = request.json
        user_input = data.get('question', '').strip()

        if not user_input:
            return jsonify({'error': '请输入内容'})

        # ============ 新增：知识点匹配 ============
        print("=" * 50)
        print(f"📝 用户问题: {user_input}")
        
        try:
            # 导入 personalization 模块（确保路径正确）
            import sys
            import os
            
            # 获取当前文件所在目录的父目录（comp4431文件夹）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            # 将父目录添加到 Python 路径
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from personalization import get_question_attention_level
            
            # 执行知识点匹配
            knowledge_path, attention_level = get_question_attention_level(user_input)
            print(f"📚 知识点路径: {knowledge_path}")
            print(f"⚠️ 注意力级别: {attention_level} (0=低, 1=中, 2=高)")
            
        except Exception as e:
            print(f"⚠️ 知识点匹配失败（不影响问答）: {str(e)}")
        
        print("=" * 50)
        # ==========================================

        # 直接转发到FastAPI后端
        response = requests.post(
            'http://localhost:8000/ask-stream',
            json={'question': user_input},
            stream=True
        )

        def generate():
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk + b'\n'

        return Response(generate(), mimetype='application/x-ndjson')

    except requests.exceptions.ConnectionError:
        return jsonify({'error': '无法连接到后端服务，请确保后端正在运行'})
    except Exception as e:
        print(f"❌ 代理请求异常: {str(e)}")
        return jsonify({'error': f'代理请求失败: {str(e)}'})

import zhipuai
import os

class ZhipuAIClient:
    def __init__(self, model="chatglm_turbo"):
        """
        初始化智谱AI客户端
        :param model: 使用的模型名称，默认为chatglm_turbo
        """
        self.model = model
        # 设置API密钥
        os.environ["ZHIPUAI_API_KEY"] = "0e3c3a2954f54436b47c73c081d4b4ca.tsPnSxLmJtQTxJNm"
        self.api_key = os.getenv("ZHIPUAI_API_KEY")

        if not self.api_key:
            raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")

        # 初始化客户端
        self.client = zhipuai.ZhipuAI(api_key=self.api_key)

    def chat(self, message):
        """
        发送消息给AI模型并获取响应
        :param message: 用户输入的消息
        :return: AI的响应内容
        """
        try:
            # 使用新的API调用方式
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                top_p=0.7
            )

            # 返回模型生成的响应
            return response.choices[0].message.content

        except Exception as e:
            return f"错误: {str(e)}"

@app.route('/generate_study_suggestion', methods=['POST'])
def generate_study_suggestion():
    """使用智谱AI生成学习建议"""
    try:
        data = request.json
        topics = data.get('topics', [])
        
        if not topics:
            return jsonify({
                'success': False,
                'message': 'No topics provided'
            })
        
        # 构建提示词
        topics_text = '\n'.join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])
        prompt = f"""Based on the following most frequently asked topics by students, provide a brief study tip (2-3 sentences maximum) to help them improve their learning:

Most frequently topic：
{topics_text}

Please provide concise, actionable recommendations in English tailored to their areas of concern, along with suggestions for next steps to deepen their understanding."""

        # 使用ZhipuAIClient生成建议
        try:
            client = ZhipuAIClient(model="glm-4")  # 使用GLM-4模型
            suggestion = client.chat(prompt)
            
            print(f"✅ 智谱AI生成建议成功")
            
            return jsonify({
                'success': True,
                'suggestion': suggestion
            })
                
        except Exception as e:
            print(f"❌ 智谱AI调用失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'智谱AI调用失败: {str(e)}'
            })
            
    except Exception as e:
        print(f"❌ 生成建议错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'服务器内部错误: {str(e)}'
        })

if __name__ == '__main__':
    # 确保数据库文件存在
    if not os.path.exists(DB_PATH):
        init_db()
    
    app.run(debug=True, port=5000)