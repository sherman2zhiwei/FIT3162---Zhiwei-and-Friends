#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 02:23:50 2019

@author: Zhiwei and Friend(s)
"""

import json
from lxml import html
import requests
import sys
import logging

def main(argv):
	# Sending request to API Link given by the server
	page = requests.get(argv[1])

	# Turn the page content into html format 
	tree = html.fromstring(page.content)

	# Get the text content (web scraping)
	html_content = tree.xpath('text()')

	# Remove newline and turn it to json format
	html_content = html_content[0].replace("\n", "")
	result_in_json = json.loads(html_content)

	# Message to validate whether this code is scraping the web content
	if list(result_in_json.keys()) == "aspects":
		logging.info("* Scraping content from API server for Aspect Term Extraction!")
	else:
		logging.info("* Scraping content from API server for Aspect Aggregation!")
	# print results found (in this case we pass back the output to the server)
	print(list(result_in_json.values())[0])

if __name__ == '__main__':
	main(sys.argv)