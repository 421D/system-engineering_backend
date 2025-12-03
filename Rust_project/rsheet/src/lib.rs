use rsheet_lib::cell_expr::{CellArgument, CellExpr};
use rsheet_lib::cell_value::CellValue;
use rsheet_lib::cells::column_number_to_name;
use rsheet_lib::command::Command;
use rsheet_lib::connect::{
    Connection, Manager, ReadMessageResult, Reader, Writer, WriteMessageResult,
};
use rsheet_lib::replies::Reply;

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::{mpsc, Arc, Condvar, Mutex}; 


//解析A1,B2等单元格坐标字符串，返回从0开始的列号和行号
fn parse_cell(s: &str) -> Option<(usize, usize)> {
    let (letters, digits): (String, String) = s.chars().partition(|c| c.is_ascii_alphabetic()); 
    if letters.is_empty() || digits.is_empty() { //如果字母或数字部分为空，返回None表示解析失败
        return None;
    }

    let row_one_based = digits.parse::<usize>().ok()?; //解析数字为行号
    let row = row_one_based - 1; //从0

    let mut col: usize = 0;

    for ch in letters.chars() {
        let v = (ch.to_ascii_uppercase() as u8 - b'A') as usize; //字母转为0-25
        col = col * 26 + (v + 1); //26进制算列号
    }
    Some((col - 1, row))
}

//根据列行，生成A1,B2单元格坐标字符串
fn key_from_rc(col: usize, row: usize) -> String {
    format!("{}{}", column_number_to_name(col as u32), row + 1) //将列号转换为字母，行号+1转数字字符串
}


fn get_cell_value_for_env(
    values: &HashMap<String, CellValue>,
    dep_err: &HashSet<String>,
    col: usize,
    row: usize,
) -> CellValue {
    let key = key_from_rc(col, row); //根据列和行 单元格的A1格式
    if dep_err.contains(&key) {
        return CellValue::Error("Depends on error".to_string());
    }
    values.get(&key).cloned().unwrap_or(CellValue::None)
}


fn get_cells(var_name: &str) -> Vec<String> { //单元格引用字符串
    if let Some((a, b)) = var_name.split_once('_') { //范围引用
        let (c1, r1) = parse_cell(a).unwrap();
        let (c2, r2) = parse_cell(b).unwrap();
        let mut v = Vec::new(); //建向量用于收集范围内所有单元格的键

        let (cl, ch) = (c1.min(c2), c1.max(c2)); //cl <= ch
        let (rl, rh) = (r1.min(r2), r1.max(r2)); //rl <= rh

        for r in rl..=rh {
            for c in cl..=ch { //每一列
                v.push(key_from_rc(c, r));
            }
        }
        return v;
    }

    let (c, r) = parse_cell(var_name).unwrap();
    vec![key_from_rc(c, r)]
}




struct Shared {  //存整个表格的数据和依赖的全局共享状态
    values: HashMap<String, CellValue>,
    exprs: HashMap<String, String>,
    versions: HashMap<String, u64>, //防止旧的计算结果覆盖新的
    depends_on_error: HashSet<String>,

    dependents: HashMap<String, HashSet<String>>, // 谁依赖我
    uses_sources: HashMap<String, HashSet<String>>, // 我依赖谁
}

impl Shared {
    fn new() -> Self { //新的空白全局共享状态
        Self {
            values: HashMap::new(),
            exprs: HashMap::new(),
            versions: HashMap::new(),
            depends_on_error: HashSet::new(),
            dependents: HashMap::new(), //空的依赖者映射
            uses_sources: HashMap::new(), //空的依赖源映射
        }
    }

   
    fn update_dep(&mut self, cell: &str, new: HashSet<String>) { //维护两个依赖图的一致性
        let cell_s = cell.to_string();

        if let Some(old) = self.uses_sources.remove(cell) {
            for s in old {
                if let Some(d) = self.dependents.get_mut(&s) { //获取该依赖源的依赖者
                    d.remove(cell); // 从依赖者集合中移除当前单元格
                    if d.is_empty() { self.dependents.remove(&s); }
                }
            }
        }

        for s in &new { //遍历新的每个依赖源
            self.dependents.entry(s.clone()).or_default().insert(cell_s.clone()); //当前单元格加入到每个新的
        }

        self.uses_sources.insert(cell_s, new);
    }

    fn get_dependents_of(&self, src: &str) -> Vec<String> { // 获取所有直接依赖给定源单元格的单元格键列表
        self.dependents.get(src).map(|s| s.iter().cloned().collect()).unwrap_or_default() // 返回依赖者集合的克隆向量，如果不存在则返回空向量
    }
}


struct WorkerState {
    pending: usize,
    completed_seq: Option<u64>,
}

struct SyncState {
    state: Mutex<WorkerState>,
    cv: Condvar,
    next_seq: Mutex<u64>,
}



fn build_scalar(var: &str, shared: &Shared) -> CellArgument { //构建Scalar类型的参数
    let (c, r) = parse_cell(var).unwrap();
    let v = get_cell_value_for_env(&shared.values, &shared.depends_on_error, c, r); //获取单元格的当前值
    CellArgument::Value(v)
}

fn build_vector(_var: &str, a: &str, b: &str, shared: &Shared) -> CellArgument {
    let (c1, r1) = parse_cell(a).unwrap();
    let (c2, r2) = parse_cell(b).unwrap();

    let (cl, ch) = (c1.min(c2), c1.max(c2));
    let (rl, rh) = (r1.min(r2), r1.max(r2));

    let mut list = Vec::new();//创建向量用于存储单元格的值

    if cl == ch {//垂直向量
        for r in rl..=rh {
            list.push(get_cell_value_for_env(&shared.values, &shared.depends_on_error, cl, r));
        }
    } else { //水平向量
        for c in cl..=ch {
            list.push(get_cell_value_for_env(&shared.values, &shared.depends_on_error, c, rl));
        }
    }

    CellArgument::Vector(list)
}


fn build_matrix(_var: &str, a: &str, b: &str, shared: &Shared) -> CellArgument { //多行多列
    let (c1, r1) = parse_cell(a).unwrap();
    let (c2, r2) = parse_cell(b).unwrap();
    let (cl, ch) = (c1.min(c2), c1.max(c2));
    let (rl, rh) = (r1.min(r2), r1.max(r2));

    let mut rows = Vec::new();

    for r in rl..=rh {
        let mut row = Vec::new();
        for c in cl..=ch {
            row.push(get_cell_value_for_env(&shared.values, &shared.depends_on_error, c, r));
        }
        rows.push(row); //当前行加入矩阵
    }

    CellArgument::Matrix(rows)
}


fn make_env(expr: &CellExpr, shared: &Shared) -> HashMap<String, CellArgument> {
    let mut env = HashMap::new();
    for var in expr.find_variable_names() {
        if let Some((a, b)) = var.split_once('_') { //范围引用
            let (c1, r1) = parse_cell(a).unwrap();
            let (c2, r2) = parse_cell(b).unwrap();
            let is_vec = r1 == r2 || c1 == c2; //是否为一维向量
            if is_vec { //如果一维
                env.insert(var.clone(), build_vector(&var, a, b, shared));
            } else { //否则二维
                env.insert(var.clone(), build_matrix(&var, a, b, shared));
            }
        } else {
            env.insert(var.clone(), build_scalar(&var, shared));
        }
    }
    env
}


fn handle_connection<R, W>( //主函数
    mut recv: R,
    mut send: W,
    shared: Arc<Mutex<Shared>>, //互斥锁
    sync: Arc<SyncState>,
    enqueue: mpsc::Sender<(String, u64)>,
) where
    R: Reader + Send + 'static,
    W: Writer + Send + 'static,
{
    loop {
        match recv.read_message() {
            ReadMessageResult::Message(msg) => {
                let maybe_reply: Option<Reply> = match msg.parse::<Command>() {
                    Ok(command) => match command { //解析成功
                        Command::Get { cell_identifier } => {
                            let key = format!( //构造单元格的A1格式键字符串
                                "{}{}",
                                column_number_to_name(cell_identifier.col as u32), //列号转字母
                                cell_identifier.row + 1
                            );
                            let get_key_clone = key.clone();
                            
                            let seq_snapshot = { //当前的序列号快照
                                let next = sync.next_seq.lock().unwrap(); //获取next_seq的互斥锁
                                *next //
                            }; //锁在此处自动释放
                            
                            if seq_snapshot > 0 {
                                let target_seq = seq_snapshot - 1;
                                let mut state = sync.state.lock().unwrap();
                                while match state.completed_seq {
                                    None => true,
                                    Some(c) => c < target_seq,
                                } {
                                    state = sync.cv.wait(state).unwrap(); //等待条件变量被唤醒，释放锁并阻塞，被唤醒后重新获取锁
                                }
                            }

                            let (is_dep_err, val) = { //从共享状态中获取单元格的值
                                let g = shared.lock().unwrap(); //获取共享状态的互斥锁
                                let error_check = g.depends_on_error.contains(&get_key_clone);
                                let value_lookup = g.values.get(&get_key_clone).cloned().unwrap_or(CellValue::None);
                                (error_check, value_lookup)
                            };
                            
                            if is_dep_err { //如果依赖错误
                                Some(Reply::Error("Depends on error".to_string()))
                            } else {
                                Some(Reply::Value(key, val))
                            }
                        }

                        Command::Set {
                            cell_identifier,
                            cell_expr,
                        } => {
                            let key = format!(
                                "{}{}",
                                column_number_to_name(cell_identifier.col as u32),
                                cell_identifier.row + 1
                            );
                            let cell_key_clone = key.clone();
                            let expr_text_clone = cell_expr.clone();//克隆表达式字符串

                            let expr = CellExpr::new(&expr_text_clone);
                            let mut new_sources: HashSet<String> = HashSet::new();
                            for var in expr.find_variable_names() { //遍历表达式中的所有变量名
                                for k in get_cells(&var) {
                                    new_sources.insert(k.clone());
                                }
                            }

                            let seq = { //SET命令分唯一的序列号
                                let mut next = sync.next_seq.lock().unwrap();
                                let s = *next;
                                *next += 1; //递增序列号下次使用
                                s
                            };

                            {
                                let mut state = sync.state.lock().unwrap(); //WorkerState的互斥锁
                                state.pending += 1;
                            }

                            { //在共享状态中更新单元格的元数据
                                let mut g = shared.lock().unwrap();

                                
                                g.exprs.insert(cell_key_clone.clone(), expr_text_clone.clone());
                                g.update_dep(&cell_key_clone, new_sources); //更新单元格的依赖关系图

                                
                                let current_v = g.versions.get(&cell_key_clone).copied().unwrap_or(0);
                                let next_v = current_v + 1; //新版本号
                                g.versions.insert(cell_key_clone.clone(), next_v); // 更新版本号
                            }

                            let _ = enqueue.send((cell_key_clone, seq));
                            None
                        }
                    },
                    Err(e) => Some(Reply::Error(e)),
                };

                if let Some(reply) = maybe_reply {
                    match send.write_message(reply) {
                        WriteMessageResult::Ok => {}
                        WriteMessageResult::ConnectionClosed => break, //连接关闭，退
                        WriteMessageResult::Err(_) => break, //出错，退
                    }
                }
            }
            ReadMessageResult::ConnectionClosed => break,
            ReadMessageResult::Err(_) => break,
        }
    }
}


fn start_worker(
    shared: Arc<Mutex<Shared>>,
    sync: Arc<SyncState>,
    rx: mpsc::Receiver<(String, u64)>,
    tx: mpsc::Sender<(String, u64)>,
) {
    while let Ok((changed, seq)) = rx.recv() {
        let (expr_str_opt, ver_snap_self) = {
            let g = shared.lock().unwrap();
            let v = g.versions.get(&changed).copied().unwrap_or(0); //当前版本号
            let ex = g.exprs.get(&changed).cloned();
            (ex, v) //表达式和版本号的元组
        };

        if let Some(expr_str) = expr_str_opt {
            let expr = CellExpr::new(&expr_str);

            let (res, is_err) = {
                let g = shared.lock().unwrap();
                let env = make_env(&expr, &g); //构建求值环境
                match expr.evaluate(&env) {
                    Ok(v) => (Some(v), false),
                    Err(_) => (None, true),
                }
            };

            {
                let mut g = shared.lock().unwrap();
                let cur = g.versions.get(&changed).copied().unwrap_or(0); //获取该单元格的当前版本号
                if cur == ver_snap_self { //版本号匹配，说明在计算期间该单元格没有被其他线程修改
                    if is_err {
                        g.depends_on_error.insert(changed.clone());
                    } else if let Some(v) = res {
                        g.depends_on_error.remove(&changed); //清除错误标记
                        g.values.insert(changed.clone(), v);
                    }
                }
            }
        }

        let deps = {
            let g = shared.lock().unwrap(); //共享状态的互斥锁
            g.get_dependents_of(&changed) //从依赖图中查找所有依赖者
        };

        if !deps.is_empty() {
            let mut state = sync.state.lock().unwrap(); //获取WorkerState的互斥锁
            for _ in &deps {
                state.pending += 1;
            }
        }

        for target in deps {
            let (expr_str, ver_snap) = {
                let g = shared.lock().unwrap();
                let v = g.versions.get(&target).copied().unwrap_or(0);
                let ex = g.exprs.get(&target).cloned();
                if let Some(s) = ex {
                    (s, v)
                } else { //如果表达式不存在
                    let mut state = sync.state.lock().unwrap(); //获取WorkerState的互斥锁
                    state.pending -= 1; //减少pending计数
                    sync.cv.notify_all();
                    continue; //跳过这个依赖者
                }
            };

            let expr = CellExpr::new(&expr_str); //创建依赖者的表达式对象


            let (res, is_err) = {
                let g = shared.lock().unwrap();
                let env = make_env(&expr, &g);
                match expr.evaluate(&env) {
                    Ok(v) => (Some(v), false),
                    Err(_) => (None, true),
                }
            };

            {
                let mut g = shared.lock().unwrap();
                let cur = g.versions.get(&target).copied().unwrap_or(0);
                if cur == ver_snap {
                    if is_err {
                        g.depends_on_error.insert(target.clone());
                    } else if let Some(v) = res {
                        g.depends_on_error.remove(&target);
                        g.values.insert(target.clone(), v);
                    }
                }
            }

            let _ = tx.send((target.clone(), seq));
        }

        
        {//原子性地更新completed_seq和pending
            let mut state = sync.state.lock().unwrap();
            state.completed_seq = Some(match state.completed_seq { //completed_seq
                None => seq,
                Some(c) => c.max(seq),
            });
            state.pending -= 1; //减少pending
            sync.cv.notify_all(); //唤醒等待的线程
        }
    }
}






pub fn start_server<M>(mut manager: M) -> Result<(), Box<dyn Error>>
where M: Manager
{
    let shared = Arc::new(Mutex::new(Shared::new()));
    let sync = Arc::new(SyncState {
        state: Mutex::new(WorkerState { // 初始化WorkerState
            pending: 0,
            completed_seq: None,
        }),
        cv: Condvar::new(),
        next_seq: Mutex::new(0),
    });

    let (tx, rx) = mpsc::channel::<(String, u64)>();
    {
        let shared_c = Arc::clone(&shared);
        let sync_c = Arc::clone(&sync);
        let tx_c = tx.clone();
        
        std::thread::spawn(move || start_worker(shared_c, sync_c, rx, tx_c));
    }

    let mut handles = Vec::new();

    loop {
        match manager.accept_new_connection() {
            Connection::NewConnection { reader, writer } => {
                let shared_c = Arc::clone(&shared);
                let sync_c = Arc::clone(&sync);
                let tx_c = tx.clone();

                handles.push(std::thread::spawn(move || {
                    handle_connection(reader, writer, shared_c, sync_c, tx_c);
                }));
            }
            Connection::NoMoreConnections => break,
        }
    }

    for h in handles { let _ = h.join(); }

    Ok(())
}