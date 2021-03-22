import pymysql, glob, time

path = "/home/rasa-stakeholder-assistant/agreements/phase2/Teaming_Agreement_test/"

def existing_teas():
    existingTEAs = []
    db = pymysql.connect('localhost', 'ebromic', 'Ericsson1', 'ai')
    try:
        cursor = db.cursor()
        sql = "SELECT Agreement_name FROM contracts_info WHERE Agreement_type = 'TEA' "
        cursor.execute(sql)
        teas = cursor.fetchall()
        print("Found {} TEAs in the database".format(str(len(teas))))
        for tea in teas:
            file_name = tea[0][:-4] + ".pdf"
            existingTEAs.append(file_name)
    except Exception as e:
        print("Error: " + e)
    return existingTEAs

def connection_to_db(teasList):
    db = pymysql.connect('localhost', 'ebromic', 'Ericsson1', 'ai')
    sql = "SELECT Agreement_Name, Digital_file FROM contracts_info WHERE Agreement_type = 'TEA' "
    # for tea in teasList:
    #     sql += "AND Agreement_Name != '{}' ".format(tea)
    try:
        cursor = db.cursor()
        cursor.execute(sql)
        teas = cursor.fetchall()
        print("TEAs found: " + str(len(teas)))
        for tea in teas:
            file_name = tea[0][:-4] + ".txt"
            file_route = path + file_name
            content = tea[1]
            with open(file_route, "a+") as f:
                f.write(content + "\n")
            # print("{} was writtern successfully...".format(file_name))
    except Exception as e:
        print("Error: " + e)

if __name__ == "__main__":
    print("Process started...")
    listTEAs = existing_teas()
    print("Size of the existing TEAs' list: {}".format(str(len(listTEAs))))
    connection_to_db(listTEAs)
    print("File writing process completed!")
