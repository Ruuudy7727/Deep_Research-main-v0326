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
import pandas as pd

host_ip="172.28.253.248"
def TD_to_API(sql):
    data={}
    data['from']="2025-01-01 00:00:00"
    data['until']="2030-01-01 00:00:00"
    url=""
    # 去除SQL中的注释
    sql = re.sub(r'--.*?\n|/\*.*?\*/', '', sql, flags=re.DOTALL)
    if 'SELECT' in sql.upper(): 
        select_part = sql[sql.upper().index('SELECT') + 7:] 
        from_index = select_part.upper().find('FROM')
        if from_index != -1: 
            select_fields = select_part[:from_index].strip().split(',') 
            data['selects'] = [field.strip() for field in select_fields]
            # 不能用 split(' ')[1]：多行 SQL 时 "FROM bmu_data\nLIMIT 10" 会得到 bmu_data\nLIMIT，表名错误导致接口非 JSON
            from_tail = select_part[from_index + 4:].lstrip()  # 跳过 "FROM"
            tab_match = re.match(r"^`?([^\s`,;`]+)`?", from_tail, re.IGNORECASE)
            tab = tab_match.group(1) if tab_match else ""
            #url=f'http://{host_ip}:8093/inner/rd-tables/{tab}/query'
            url=f'http://{host_ip}:8093/inner/td-tables/{tab}/query' if tab else ""
    if 'WHERE' in sql.upper():
        where_pattern = r'WHERE\s+(.*?)(?:\s+ORDER|\s+LIMIT|$)'
        where_match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            data['whereSql'] = where_clause
            # 解析 ts BETWEEN '...' AND '...'
            ts_between = re.search(
                r'\bts\s+BETWEEN\s+[\'"]([^\'"]+)[\'"]\s+AND\s+[\'"]([^\'"]+)[\'"]',
                where_clause, re.IGNORECASE
            )
            if ts_between:
                data['from']  = ts_between.group(1).strip()
                data['until'] = ts_between.group(2).strip()
            else:
                # 解析 ts >= / ts > (from)
                ts_from = re.search(
                    r'\bts\s*(>=|>)\s*[\'"]([^\'"]+)[\'"]',
                    where_clause, re.IGNORECASE
                )
                if ts_from:
                    data['from'] = ts_from.group(2).strip()
                # 解析 ts <= / ts < (until)
                ts_until = re.search(
                    r'\bts\s*(<=|<)\s*[\'"]([^\'"]+)[\'"]',
                    where_clause, re.IGNORECASE
                )
                if ts_until:
                    data['until'] = ts_until.group(2).strip()
    # 解析ORDER BY子句
    if 'ORDER BY' in sql.upper():
        order_pattern = r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)'
        order_match = re.search(order_pattern, sql, re.IGNORECASE | re.DOTALL)
        if order_match:
            order_content = order_match.group(1).strip()
            # 分割排序字段
            order_fields = re.split(r',(?![^\(]*\))', order_content)
            
            for field in order_fields:
                field = field.strip()
                # 检查排序方向
                if field.upper().endswith(' DESC'):
                    data['order']='desc'
                else:
                    data['order']='asc'
    # 解析LIMIT子句
    if 'LIMIT' in sql.upper(): 
        limit_pattern = r'LIMIT\s+(\d+)(?:\s*,?\s*(\d+))?'
        limit_match = re.search(limit_pattern, sql.upper(), re.IGNORECASE)
        
        if limit_match:
            if limit_match.group(2):
                # LIMIT m, n 格式
                data['limit'] = [int(limit_match.group(1)), int(limit_match.group(2))]
            else:
                # LIMIT n 格式
                data['limit'] = int(limit_match.group(1)) #[0, int(limit_match.group(1))]
    else:
        data['limit'] = 2000
    # data['whereSql']="bms_code='00256001001'"
    return url,data

def getTdSqlData(sql,asname={}, debug=False):
    url, data = TD_to_API(sql)
    if debug:
        print("[TD DEBUG] url:", url)
        print("[TD DEBUG] payload:", data)
    if not url:
        print("数据接口解析获取数据失败! (无法解析出表名，请检查 SQL 的 FROM 子句)")
        return pd.DataFrame()
    try:
        rs = requests.post(url, json=data, timeout=60)
        if debug:
            print("[TD DEBUG] status_code:", getattr(rs, "status_code", "?"))
            print("[TD DEBUG] response_preview:", (rs.text or "")[:500])
    except Exception as e:
        print(f"数据接口请求失败! {e}")
        return pd.DataFrame()
    # print(rs.text)
    try:
        data = pd.DataFrame(json.loads(rs.text))
        if asname != {}:
            data.rename(columns=asname, inplace=True)
    except Exception:
        print("数据接口解析获取数据失败!")
        print("HTTP", getattr(rs, "status_code", "?"), (rs.text or "")[:1500])
        data = pd.DataFrame()
    return data
