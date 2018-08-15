學號：R05943092
姓名：曾育為
編譯與執行 : python hw1.py -[E|A|W][l|b|p]
Ex:python hw1.py -El -Ab -Wp
程式內容說明:
main()  -> 主程式、處理argv參數 
parse() -> 用urllib.request去讀取資料，儲存於table中
process(table,args[1]) 	-> 將資料進行分析處理(male/female/total)，儲存於output_data中
plot_line(table,args[1])-> 將output_data，用matplotlib畫出折線圖
plot_bar(table,args[1])	-> 將output_data，用matplotlib畫出柱狀圖
plot_pie(table,args[1])	-> 將output_data，用matplotlib畫出圓餅圖