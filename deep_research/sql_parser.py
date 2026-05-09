# -*- coding: utf-8 -*-
'''
Created on 2025年10月29日
解析sql，查询内部定义的mysql 云档案的接口查询
测试环境：host_ip="172.28.253.241"
from sql_parser import getMySqlData
使用 getMySqlData(sql,asname={}) 调用，返回dataframe
@author: thankusun
'''
import requests
import json
import re
import os
import pandas as pd

host_ip = "172.28.253.241"
SQL_PARSER_VERBOSE = int(os.getenv("SQL_PARSER_VERBOSE", "1") or "1")


def _clean_sql(sql):
    """清理SQL：去注释、去多余空白，返回单行干净SQL"""
    # 去单行注释
    sql = re.sub(r'--[^\n]*', '', sql)
    # 去多行注释
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # 所有空白符（换行\n、制表\t、多空格）统一替换为单空格
    sql = re.sub(r'\s+', ' ', sql).strip()
    # 去末尾分号
    sql = sql.rstrip(';').strip()
    return sql


def mysql_to_API(sql):
    data = {}
    # ✅ 默认宽泛时间范围，避免查不到数据
    data['from'] = "2000-01-01 00:00:00"
    data['until'] = "2099-12-31 23:59:59"
    url = ""

    # ✅ 第一步：清理SQL，消除所有换行符影响
    sql = _clean_sql(sql)
    if SQL_PARSER_VERBOSE >= 2:
        print(f"[DEBUG] 清理后SQL: {sql}")

    # ✅ 解析 SELECT 字段 和 表名
    select_match = re.search(
        r'SELECT\s+(.*?)\s+FROM\s+(\w+)',
        sql,
        re.IGNORECASE
    )
    if select_match:
        fields_str = select_match.group(1).strip()
        tab = select_match.group(2).strip()
        data['selects'] = [f.strip() for f in fields_str.split(',') if f.strip()]
        url = f'http://{host_ip}:8093/inner/rd-tables/aiops_cloud/{tab}/query'
        if SQL_PARSER_VERBOSE >= 2:
            print(f"[DEBUG] 表名: [{tab}]")
            print(f"[DEBUG] 字段: {data['selects']}")
    else:
        raise ValueError("SQL解析失败：无法匹配 SELECT...FROM 结构，请检查SQL语句")

    # ✅ 解析 WHERE 子句
    where_match = re.search(
        r'\bWHERE\b\s+(.*?)(?:\s+\bORDER\s+BY\b|\s+\bLIMIT\b|$)',
        sql,
        re.IGNORECASE
    )
    if where_match:
        data['whereSql'] = where_match.group(1).strip()
        if SQL_PARSER_VERBOSE >= 2:
            print(f"[DEBUG] WHERE: {data['whereSql']}")

        # ✅ 核心修复：从WHERE中提取时间范围，赋值给from/until
        # 匹配 start_time >=/<= '2025-03-19 00:00:00' 格式
        time_pattern = r'start_time\s*(>=|<=|>|<|=)\s*[\'"](\d{4}-\d{2}-\d{2}[\s\d:]*)[\'"]'
        time_matches = re.findall(time_pattern, data['whereSql'], re.IGNORECASE)

        from_time = None
        until_time = None

        for op, t in time_matches:
            t = t.strip()
            if op in ('>=', '>'):
                from_time = t
            elif op in ('<=', '<'):
                until_time = t
            elif op == '=':
                from_time = t
                until_time = t

        if from_time:
            data['from'] = from_time
            if SQL_PARSER_VERBOSE >= 2:
                print(f"[DEBUG] FROM 提取自WHERE: {data['from']}")
        if until_time:
            data['until'] = until_time
            if SQL_PARSER_VERBOSE >= 2:
                print(f"[DEBUG] UNTIL 提取自WHERE: {data['until']}")

    # ✅ 解析 ORDER BY 子句
    order_match = re.search(
        r'\bORDER\s+BY\b\s+(.*?)(?:\s+\bLIMIT\b|$)',
        sql,
        re.IGNORECASE
    )
    if order_match:
        order_content = order_match.group(1).strip()
        data['order'] = 'desc' if re.search(r'\bDESC\b', order_content, re.IGNORECASE) else 'asc'
        if SQL_PARSER_VERBOSE >= 2:
            print(f"[DEBUG] ORDER: {data['order']}")

    # ✅ 解析 LIMIT 子句（支持 LIMIT n 和 LIMIT offset,n）
    limit_match = re.search(r'\bLIMIT\b\s+(\d+)\s*(?:,\s*(\d+))?', sql, re.IGNORECASE)
    if limit_match:
        if limit_match.group(2):
            data['offset'] = int(limit_match.group(1))
            data['limit'] = int(limit_match.group(2))
        else:
            data['limit'] = int(limit_match.group(1))
        if SQL_PARSER_VERBOSE >= 2:
            print(f"[DEBUG] LIMIT: {data.get('limit')}")

    return url, data


def _coerce_columnar_dict_to_rows(payload: dict):
    """将 {col:[...], col2:[...]} 转为 [{...}, {...}]。"""
    if not isinstance(payload, dict) or not payload:
        return None
    vals = list(payload.values())
    if not vals:
        return None
    if not all(isinstance(v, list) for v in vals):
        return None
    row_count = max((len(v) for v in vals), default=0)
    if row_count <= 0:
        return []
    rows = []
    for i in range(row_count):
        row = {}
        for k, v in payload.items():
            row[k] = v[i] if i < len(v) else None
        rows.append(row)
    return rows


def _normalize_api_payload(parsed: object):
    """
    兼容接口返回：
    1) list[dict]
    2) {"data": list[dict]}
    3) {"data": {col:[...]}}（列式）
    4) {col:[...]}（列式）
    """
    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict) and "data" in parsed:
        data_obj = parsed.get("data")
        if isinstance(data_obj, list):
            return data_obj
        if isinstance(data_obj, dict):
            rows = _coerce_columnar_dict_to_rows(data_obj)
            if rows is not None:
                if SQL_PARSER_VERBOSE >= 1:
                    print(f"[sql_parser] 检测到列式 data，已转换为行式: {len(rows)} 行")
                return rows

    if isinstance(parsed, dict):
        rows = _coerce_columnar_dict_to_rows(parsed)
        if rows is not None:
            if SQL_PARSER_VERBOSE >= 1:
                print(f"[sql_parser] 检测到列式响应，已转换为行式: {len(rows)} 行")
            return rows
        return [parsed]

    return [parsed]


def getMySqlData(sql, asname={}):
    # ✅ SQL解析
    try:
        url, data = mysql_to_API(sql)
    except ValueError as e:
        print(f"[sql_parser] SQL解析错误: {e}")
        return pd.DataFrame()

    if not url:
        print("[sql_parser] 无法生成请求URL，请检查SQL语句")
        return pd.DataFrame()

    print(f"\n[sql_parser] 请求URL: {url}")
    if SQL_PARSER_VERBOSE >= 2:
        print(f"[sql_parser] 请求参数: {json.dumps(data, ensure_ascii=False, indent=2)}\n")
    else:
        print(
            "[sql_parser] 请求摘要: "
            f"from={data.get('from')} until={data.get('until')} "
            f"selects={len(data.get('selects', []))} "
            f"limit={data.get('limit')} order={data.get('order', '-')}"
        )

    payload = dict(data)
    max_retry = max(1, len(payload.get("selects", [])) + 1)
    last_error = None

    for attempt in range(max_retry):
        # ✅ 发起请求
        try:
            rs = requests.post(url, json=payload, timeout=30)
        except requests.exceptions.Timeout:
            last_error = "请求超时(30s)"
            print(f"[sql_parser] {last_error}")
            break
        except requests.exceptions.ConnectionError as e:
            last_error = f"网络连接失败: {e}"
            print(f"[sql_parser] {last_error}")
            break

        # ✅ 解析响应
        try:
            parsed = json.loads(rs.text)
        except Exception as e:
            last_error = f"响应非JSON格式: {e}\n原始响应: {rs.text[:200]}"
            print(f"[sql_parser] {last_error}")
            break

        # ✅ 接口报错处理
        if isinstance(parsed, dict) and parsed.get("error_code"):
            err_code = str(parsed.get("error_code", ""))
            err_desc = str(parsed.get("error_desc", ""))
            last_error = f"error_code={err_code}, error_desc={err_desc}"
            print(f"[sql_parser] 接口报错(第{attempt+1}次): {last_error}")

            # ✅ 自动移除非法字段并重试
            if err_code == "param_invalid" and "selects[" in err_desc:
                m = re.search(r"selects\[\d+\]\s*=\s*([A-Za-z0-9_]+)", err_desc)
                bad_field = m.group(1) if m else ""
                if bad_field and bad_field in payload.get("selects", []):
                    payload["selects"] = [f for f in payload["selects"] if f != bad_field]
                    print(f"[sql_parser] 自动移除非法字段: {bad_field}，剩余: {payload['selects']}")
                    if payload["selects"]:
                        continue
                    else:
                        last_error = "所有字段均被移除"
            break

        # ✅ 正常数据处理
        try:
            normalized_rows = _normalize_api_payload(parsed)
            df = pd.DataFrame(normalized_rows)

            if asname:
                df.rename(columns=asname, inplace=True)

            print(f"[sql_parser] ✅ 成功获取数据：{len(df)} 行 × {len(df.columns)} 列")
            return df

        except Exception as e:
            last_error = f"DataFrame构建失败: {e}"
            print(f"[sql_parser] {last_error}")
            break

    print(f"\n[sql_parser] ❌ 数据获取失败: {last_error or 'unknown error'}")
    return pd.DataFrame()


if __name__ == "__main__":
    test_sql = """
    SELECT
        id,
        station_code,
        bmu_id,
        bmu_code,
        cell_id,
        grade,
        average_severity,
        summary_cn,
        start_time,
        end_time
    FROM alarm_event
    WHERE station_code = 'station-00256'
        AND start_time >= '2025-03-19 00:00:00'
        AND start_time <= '2025-03-23 23:59:59'
    ORDER BY start_time DESC
    LIMIT 50;
    """
    df = getMySqlData(test_sql)
    print(df)
