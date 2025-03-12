#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:PythonHub
@File:backend.py.py
@IDE:PythonHub
@Author:whz
@Date:2025/3/12 11:44

'''
import sqlite3
import ast
import json


class SearchEngine:
    #创建一个IdToDoc的表，整数主键 id 列，文本 document列用于存储文档内容
    #创建一个WordToId的表，文本 name 列，文本 value 列。用于存储单词和对应的文档ID信息。
    def __init__(self):
        self.conn = sqlite3.connect("searchengine.sqlite3",autocommit=True)
        cur = self.conn.cursor()
        res = cur.execute("SELECT NAME FROM sqlite_master WHERE name='IdToDoc'")
        tables_exist = res.fetchone()
        if not tables_exist:
            self.conn.execute("CREATE TABLE IdToDoc(id INTEGER PRIMARY KEY, document TEXT)")
            self.conn.execute('CREATE TABLE WordToId (name TEXT, value TEXT)')
            cur.execute("INSERT INTO WordToId VALUES (?, ?)", ("index", "{}",))

    def index_document(self,document):
        row_id = self._add_to_IdToDoc(document)
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        document = document.split()
        for word in document:
            if word not in reverse_idx:
                reverse_idx[word] = [row_id]
            else:
                if row_id not in reverse_idx[word]:
                    reverse_idx[word].append(row_id)
        reverse_idx = json.dumps(reverse_idx)
        cur = self.conn.cursor()
        result = cur.execute("UPDATE WordToId SET value = (?) WHERE name = 'index'",(reverse_idx,))
        return("index successful")


    def _add_to_IdToDoc(self,document):
        cur = self.conn.cursor()
        res = cur.execute("INSERT INTO IdToDoc (document) VALUES (?)", (document,))
        return res.lastrowid

    def _find_documents_with_idx(self, idxs):
        idxs = list(idxs)
        cur = self.conn.cursor()
        sql="SELECT document FROM IdToDoc WHERE id in ({seq})".format(
                                                                seq=','.join(['?']*len(idxs))
                                                               )
        result = cur.execute(sql, idxs).fetchall()
        return(result)

    def find_documents(self, search_term):
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        search_term = search_term.split(" ")
        all_docs_with_search_term = []
        for term in search_term:
            if term in reverse_idx:
                all_docs_with_search_term.append(reverse_idx[term])

        if not all_docs_with_search_term: # the search term does not exist
            return []

        common_idx_of_docs = set(all_docs_with_search_term[0])
        for idx in all_docs_with_search_term[1:]:
            common_idx_of_docs.intersection_update(idx)

        if not common_idx_of_docs: # the search term does not exist
            return []

        return self._find_documents_with_idx(common_idx_of_docs)
if __name__ == "__main__":
    se = SearchEngine()
    se.index_document("we should all strive to be happy and happy again")
    print(se.index_document("happiness is all you need"))
    se.index_document("no way should we be sad")
    se.index_document("a cheerful heart is a happy one even in Nigeria")
    print(se.find_documents("happy"))