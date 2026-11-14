#!/bin/sh
# Regenerates this report. Input file sha256: d8c03863d91d3a61a6249bd849f13e4f5f48092e7fd99c22f93bbdc3554e1af6
ads report examples/orders.csv --out demo/orders --question 'top 2 regions by total revenue' --date-col order_date --value-col revenue --horizon 6 --seed 0 --problem 'Where does revenue come from, and what range should we plan for over the next six months?'
