import sqlite3

class Grade_converter:

    def __init__(self):
        self.db_path = r'../../data/raw/db.sqlite3'
        


    def get_grade_dictionary(self):

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

    
        query = """
        SELECT
            difficulty, 
            boulder_name
        FROM difficulty_grades
        """

        cursor.execute(query)
        result = cursor.fetchall()

        grade_dict = {}

        for x in range(len(result)):
            grade_dict[(result[x][0])] = result[x][1].split("/")[1]

        return grade_dict
