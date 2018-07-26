""" parses text and segmentation data from IAM xml files """

from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
import os

# utility functions

remove_extension = lambda f: os.path.splitext(f)[0]

def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# init

src_dir = "xml"
out_dir = "IAM_form_data"

mkdir(out_dir)

# main parsing loop

for file in os.listdir(src_dir):
    form_id = remove_extension(file)

    with open(os.path.join(src_dir, file), "r") as xml:
        # reads and parses all data
        raw = xml.read()
        root = BeautifulSoup(raw, "html5lib")
        writer = root.form["writer-id"]

        mkdir(os.path.join(out_dir, writer))
        
        lines = root.form.find("handwritten-part").find_all("line")
        html = HTMLParser()
        line_list = []
        line_id_list = []
        for line in lines:
            line_list.append(html.unescape(line["text"]))
            line_id_list.append(line["id"])
        
        # saves all forms' text
        form_text = '\n'.join(line_list)

        with open(os.path.join(out_dir, writer, form_id) + ".form", "w") as f:
            f.write('\n'.join(line_list))

        # calculates and saves word segment indices
        word_data = []
        curr_offset = 0

        err = False
        for word in root.form.find("handwritten-part").find_all("word"):
            word_start = form_text.find(word["text"], curr_offset)
            if word_start == -1:
                print file
                err = True
                break
            word_end = word_start + len(word["text"])
            word_data.append({"id": word["id"], "start": word_start, "end": word_end})
            curr_offset = word_end

        if err:
            continue

        with open(os.path.join(out_dir, writer, form_id) + ".words", "w") as f:
            for word in word_data:
                f.write("{id},{start},{end}\n".format(id=word["id"], start=word["start"], end=word["end"]))

        # calculates and saves line segment indices
        with open(os.path.join(out_dir, writer, form_id) + ".lines", "w") as f:
            line_start = 0
            for line, _id in zip(line_list, line_id_list):
                line_end = line_start+len(line)
                f.write("{id},{start},{end}\n".format(id=_id, start=line_start, end=line_end))
                line_start = line_end+1

