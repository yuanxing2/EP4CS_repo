
for program_type in ["Java", "Python"]:
    for type_data in ["train", "test"]:
        path1 = program_type + "_data/"+ type_data +"_id_type1.txt"
        with open(path1, 'r') as f1:
            res1 = f1.read()
            res1 = res1.split("##########")
            f1.close()
        if program_type == "Python":
            pass
        else:
            path2 = program_type + "_data/"+ type_data +"_id_type2.txt"
            with open(path2, 'r') as f2:
                res2 = f2.read()
                res2 = res2.split("##########")
        path3 = program_type + "_data/"+ type_data +"_id_type3.txt"
        with open(path3, 'r') as f3:
            res3 = f3.read()
            res3 = res3.split("##########")
        if program_type == "Python":
            pass
        else:
            path4 = program_type + "_data/"+ type_data +"_id_type4.txt"
            with open(path4, 'r') as f4:
                res4 = f4.read()
                res4 = res4.split("##########")
        path5 = program_type + "_data/"+ type_data +"_repo.txt"
        with open(path5, 'r') as f5:
            res5 = f5.read()
            res5 = res5.split("##########")
        path6 = program_type + "_data/"+ type_data +"_dfg.txt"
        with open(path6, 'r') as f6:
            res6 = f6.read()
            res6 = res6.split("##########")

        example = []
        for i in range(len(res1)-1):
            temp = ""
            if program_type == "Python":
                pass
            else:
                temp = res2[i][res2[i].index("\n") + 1:]
            temp = temp + "Identifiers: " + res1[i][res1[i].index("\n") + 1:]
            repo = res3[i][res3[i].index("\n") + 1:]
            temp = temp + repo[repo.index("\n") + 1:]
            if program_type == "Python":
                pass
            else:
                fun_name = res4[i][res4[i].index("\n") + 1:]
            temp = temp + fun_name[fun_name.index("\n") + 1:]
            temp = temp + res5[i][res5[i].index("\n") + 1:]
            temp = temp + "DFG: " + res6[i][res6[i].index("\n") + 1:]
            example.append(temp)


        with open(program_type + "_data/background_knowledge_"+ type_data +".txt", 'w') as fg:
            fg.write("##########\n".join(example))
            fg.close()
