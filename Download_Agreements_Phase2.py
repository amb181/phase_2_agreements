import pymysql

agreement_type = "NDA"
path = "/home/ealloem/Documents/Phase2_agreements/" + agreement_type + "/"

def ConnectDB (sql):
    db = pymysql.connect('localhost', 'ealloem', 'Ericsson1', 'ai')
    cursor = db.cursor()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except Exception as e:
        print("Error: " + e)
    cursor.close()

    return results


def Existing_Agreements():
    existing = []
    sql = "SELECT Agreement_name FROM " + agreement_type + ";"
    results = ConnectDB(sql)
    print("Found {} {}s in the server".format(len(results), agreement_type))
    for result in results:
        file_name = result[0]
        existing.append(file_name)
        
    return existing


def Download_Agreements(agreements_list):
    sql = "SELECT DISTINCT Agreement_Name, Digital_file FROM contracts_info WHERE Agreement_type = '" + agreement_type + "'"
    if len(agreements_list) != 0:
        for agreement in agreements_list:
            sql += " AND Agreement_Name != '{}' ".format(agreement)

    results = ConnectDB(sql)
    print("{}s found: {}".format(agreement_type, len(results)))

    for result in results:
        file_name = result[0]
        file_name = file_name.replace(".pdf", ".txt")
        file_route = path + file_name
        content = result[1]

        with open(file_route, "w+") as f:
            f.write(content + "\n")


if __name__ == "__main__":
    print("Process started...")
    list_existing = Existing_Agreements()
    Download_Agreements(list_existing)
    print("File writing process completed!")