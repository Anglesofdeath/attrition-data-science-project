import numpy as np
import yaml
from math import isnan

def returnAbsoluteValueMonthlyIncome(income: int):
    '''
    Takes an integer input and returns the absolute value
    '''
    return abs(income)

def ordinalEncodingQualifications(qualifications: str):
    '''
    Does cleaning and ordinal encoding for qualifications feature
    '''
    if qualifications == 'Diploma':
        return 0
    elif qualifications == "Bachelor's" or qualifications == 'Bachelor':
        return 1
    elif qualifications == "Master's" or qualifications == 'Master':
        return 2
    elif qualifications == 'Doctor of Philosophy' or qualifications == 'Ph.D':
        return 3

def ordinalEncodingMembership(membership: str):
    '''
    Does ordinal encoding for membership feature
    '''
    if membership == 'Normal':
        return 0
    elif membership == 'Bronze':
        return 1
    elif membership == 'Silver':
        return 2
    elif membership == 'Gold':
        return 3

def reduceBranch(branch: str):
    '''
    Combines the Changi and Kranji categories of the Branch feature into one category
    '''
    if branch == 'Changi' or branch == 'Kranji':
        return 'Changi & Kranji'
    else:
        return branch

def convertYearToClientAge(birth_year :int):
    return 2022 - birth_year

def fillInMissingBirthYearValues(join_age:float, birth_year: int, months_member:int):
    if isnan(birth_year):
        return 2022 - np.round(join_age + (months_member/12))
    else:
        return birth_year

def fillInMissingAgeValues(join_age: float, current_age:float, months_member: int):
    if isnan(join_age):
        return np.round(current_age - (months_member/12))
    else:
        return join_age

def changeCategoryOfWorkDomain(work_domain:str):
    if work_domain == 'Medical' or work_domain == 'Engineering' or work_domain == 'Information Technology':
        return 'Med, Eng & IT'
    elif work_domain == 'Business' or work_domain == 'Science':
        return 'Bus & Sci'
    elif work_domain == 'Unemployed':
        return work_domain

def createWeeklyHours(usage_rate:int, usage_time:float):
    return usage_rate * usage_time

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)



