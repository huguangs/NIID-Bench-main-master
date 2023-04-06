import xlwt

wb = xlwt.Workbook()
# 添加一个表
ws = wb.add_sheet('test')
# 3个参数分别为行号，列号，和内容
# 需要注意的是行号和列号都是从0开始的

# ws.write(0, 0, '第1列')   # 一个一个来
# ws.write(0, 1, '第2列')
# ws.write(0, 2, '第3列')
for i in range(3):
    for j in range(2):
        ws.write(i, j, f'第{i + 1}行,第{j + 1}列')

# 保存excel文件
wb.save('./test.xls')