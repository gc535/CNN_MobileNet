labels = open("labels.txt", "r")
str_to_write = "{"
for line in labels:
	col_ind = line.index(":")
	pred_class = line[col_ind+1:]
	str_to_write += '"' + pred_class.strip("\n") + '"' + ','
str_to_write+='}'
text_file = open("output.txt", "w")
text_file.write(str_to_write)
text_file.close()