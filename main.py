from flask import Flask
from app import views

app = Flask(__name__)

# url
app.add_url_rule('/','fresh',views.fresh,methods=['GET','POST'])
# 
if __name__ == "__main__":
    app.run(debug=True)
    