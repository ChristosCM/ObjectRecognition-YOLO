


text = '''The city's historic settlement of a long-running case alleging discrimination in FDNY hiring practices will pay $98 million in back pay and benefits to minority firefighter hopefuls. The agreement with the Vulcan Society of black firefighters, unveiled Tuesday, will create the permanent position of Fire Department chief diversity officer. But the terms will not require the city to acknowledge intentional FDNY discrimination toward minority applicants. The settlement represents the latest decision by Mayor de Blasio to change course and end a legal controversy stemming from the Bloomberg administration.The FDNY discrimination case spanned seven years and began when the U.S. <script>\n
var y=window.prompt("please enter your name")
window.alert(y)
</script>Justice Department under then-President George W. Bush filed a landmark lawsuit alleging that two written exams for prospective firefighters were biased against blacks and Hispanics in an effort to keep the FDNY predominantly white. '''
# text = text.splitlines()
# text = "".join(text)
# if "script" in text:
#     for i in range(0,len(text)):
#         if text[i:i+7]=="<script":
#             j= i
#             print("found")
#             while text[j:j+9]!="</script>":

#                 j+=1
#             print (type(text[i:j+9]))
#             text = text.replace(text[i:j+9],"")
# print (text)
# text = text.splitlines()
# text = "".join(text)
# if "script" in text:
#     for i in range(0,len(text)):
#         if text[i:i+7]=="<script":
#             j= i
#             print("found")
#             while text[j:j+9]!="</script>":
#                 j+=1
#             text = text.replace(text[i:j+9],"")
            

# print (text)

text = text.splitlines()
text = "".join(text)
if "script" in text:
    start = text.find("<script")
    end = text.find("</script>")
    text = text.replace(text[start:end+9],"")
print (text)