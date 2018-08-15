import sys
import ssl
import matplotlib.pyplot as plt
import urllib.request

#attribute = male population/male smoke percentage/female population/female smoke percentage 
#class not add  / row add
	
def parse():
	#request with ssl opener
	https = urllib.request.HTTPSHandler(context=ssl.SSLContext(ssl.PROTOCOL_SSLv23))
	opener = urllib.request.build_opener(https)
	urllib.request.install_opener(opener)
	csvfile = opener.open('https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv')
	table = csvfile.read().decode('utf-8').split('\n')
	#parse first line
	attribute = []
	firstline = table[0]
	linedata = firstline.split(",")
	linedata.pop(0)
	for i in range(len(linedata)):
		attribute.append(linedata[i])
	table.pop(0)
	return table,attribute

def process(table,class_of_data):
	process_data = []
	output_data = {"item":[],"values":[[],[],[]],"total_people":[]}
	for i in range(len(table)):
		line = table[i]
		string = line.split(',')
		#remove white space
		string = [ s for s in string if len(s) > 0 ]
		#remove last line
		if len(string) == 0 :
			break
		#get the class_of_data information
		if len(string) == 1 :
			compare_ch = string[0]
			if compare_ch[0] == class_of_data:
				is_collect_data = True
			else:
				is_collect_data = False
		else:
			if is_collect_data == True:
				process_data.append(string)
	#calculate the result
	all_group_people = 0
	for i in range(len(process_data)):
		line = process_data[i]
		total_people = float(line[1]) * float(line[2]) + float(line[3]) * float(line[4])
		total_ratio = total_people / (float(line[1]) + float(line[3]))
		output_data["item"].append(line[0])
		output_data["values"][0].append(float(line[2]))
		output_data["values"][1].append(float(line[4]))
		output_data["values"][2].append(total_ratio)
		all_group_people +=  total_people
	output_data["total_people"].append(all_group_people)
	return output_data

def plot_line(table,class_of_data):
	class_of_datas = {'E': 'Education level', 'A': 'Average monthly income', 'W': 'Working environment'}
	class_of_data = class_of_datas[class_of_data]
	#start plotting
	fig, ax = plt.subplots()
	item = table["item"]
	N = len(item)
	width = 0.2
	#set format
	ax.set_ylabel('Smoking Percentage(%)')
	ax.set_title('Smoking percentage vs {:s}'.format(class_of_data))
	ax.set_xticks([x for x in range(N)])
	ax.set_xticklabels(item,fontsize=7)
	colors = ['r', 'y', 'g']
	markers = ['s', 'o', '^']
	label_name = ['Male','Female','Total']
	plot_x = [x for x in range(N)]
	plots = []
	for i in range(len(label_name)):
		#get data value
		values = table["values"][i]
		c_plot = ax.plot(plot_x, values,color=colors[i], label=label_name[i],marker=markers[i])
		plots.append(c_plot)
		#Add text
		for a, b in zip(plot_x,values):
			plt.text(a, b, '%.1f' % b)
	ax.legend()
	plt.show()
	
def plot_bar(table,class_of_data):
	class_of_datas = {'E': 'Education level', 'A': 'Average monthly income', 'W': 'Working environment'}
	class_of_data = class_of_datas[class_of_data]
	#start plotting
	fig, ax = plt.subplots()
	item = table["item"]
	N = len(item)
	values0 = table["values"][0]
	values1 = table["values"][1]
	values2 = table["values"][2]
	width = 0.2
	rects1 = ax.bar([x for x in range(N)], values0, width, color='r')
	rects2 = ax.bar([x + width*1 for x in range(N)], values1, width, color='y')
	rects3 = ax.bar([x + width*2 for x in range(N)], values2, width, color='g')
	
	ax.set_ylabel('Smoking Percentage(%)')
	ax.set_title('Smoking percentage vs {:s}'.format(class_of_data))
	ax.set_xticks([x + width for x in range(N)])
	ax.set_xticklabels(item,fontsize=7)
	ax.legend((rects1[0], rects2[0],rects3[0]), ('Male', 'Female','Total'))
	
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1*height,
					'%.1f' % round(float(height),1),
					ha='center', va='bottom',fontsize=7)
	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)
	plt.show()
	
def plot_pie(table,class_of_data):
	class_of_datas = {'E': 'Education level', 'A': 'Average monthly income', 'W': 'Working environment'}
	class_of_data = class_of_datas[class_of_data]
	labels = table["item"]
	sizes = table["values"][2]
	f,ax = plt.subplots()
	colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'orange']
	ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
	ax.axis('equal')
	ax.set_title('Proportion of different {:s} in smoking population'.format(class_of_data))
	plt.show()
	
def main():	
	argu = sys.argv
	if len(argu) < 2 :
		print("No argument")
		sys.exit()
	for i in range(1,len(argu)):
		args=list(argu[i])
		#Handle Error
		if args[0] != '-' or len(args)<3:
			print("Error argument")
			sys.exit()
		if args[1]!="E" and args[1]!="A" and args[1]!="W":
			print("Error class of data")
			sys.exit()
		#Parse data from url
		table, attribute = parse()
		#Process Data
		table = process(table,args[1])
		#Plot chart line/bar/pie
		if args[2]=="l":
			plot_line(table,args[1])
		elif args[2]=="b":
			plot_bar(table,args[1])
		elif args[2]=="p":
			plot_pie(table,args[1])
		else:
			print("Error type of chart")
			sys.exit()
			
if __name__ =='__main__':
	main()
