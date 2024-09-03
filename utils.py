import pandas as pd
from lxml import etree
import re
from collections import defaultdict

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def read_posts(file_path):
  """
  Reads a given xml file and converts it into a pandas DataFrame.

  This function parses the given xml file, extracts relevant information
  and returns a pandas DataFrame. The DataFrame's columns are as follows:

  - Id
  - PostTypeId
  - ParentId
  - AcceptedAnswerId
  - CreationDate
  - Score
  - ViewCount
  - Body (cleaned of html)
  - OwnerUserId
  - LastActivityDate
  - Title
  - Tags
  - AnswerCount
  - CommentCount
  - ContentLicense

  :param file_path: The path to the xml file to be parsed.
  :type file_path: str

  :return: A pandas DataFrame containing the parsed data.
  :rtype: pandas.DataFrame
  """
  d = defaultdict(list)

  doc = etree.parse(file_path)

  rows = doc.xpath('.//row')

  for r in rows:
    d['Id'].append(r.get('Id'))
    d['PostTypeId'].append(r.get('PostTypeId'))
    d['ParentId'].append(r.get('ParentId'))
    d['AcceptedAnswerId'].append(r.get('AcceptedAnswerId'))
    d['CreationDate'].append(r.get('CreationDate'))
    d['Score'].append(r.get('Score'))
    d['ViewCount'].append(r.get('ViewCount'))

    #for human readiblity
    raw_body_text = r.get('Body')
    clean_body_text = clean_html(raw_body_text)
    d['Body'].append(clean_body_text)

    d['OwnerUserId'].append(r.get('OwnerUserId'))
    d['LastActivityDate'].append(r.get('LastActivityDate'))
    d['Title'].append(r.get('Title'))
    d['Tags'].append(r.get('Tags'))
    d['AnswerCount'].append(r.get('AnswerCount'))
    d['CommentCount'].append(r.get('CommentCount'))
    d['ContentLicense'].append(r.get('ContentLicense'))

  df = pd.DataFrame(d)
  return df

def get_qas(df,leave_id_out=True):
  """
  Cleans dataframes for later usage

    Args:
        df (DataFrame): The given DataFrame.
        leave_id_out (bool): Boolean switch for the id column.

    Returns:
        DataFrame: the cleaned DataFrame.
  """
  answers = df[df['ParentId'].notna()]
  questions = df[df['ParentId'].isna()]

  # drop unnecessary columns
  q = questions.drop(columns=['ParentId','PostTypeId','OwnerUserId','CommentCount','ContentLicense','CreationDate','LastActivityDate','ViewCount'])
  a = answers.drop(columns=['PostTypeId','OwnerUserId','CommentCount','ContentLicense','Title','Tags','AnswerCount','ViewCount','AcceptedAnswerId','CreationDate','LastActivityDate'])

  # merge
  merged = pd.merge(q, a, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))
  merged = merged[['Id_question', 'Body_question', 'Body_answer', 'Score_answer']]

  merged.columns = [ 'question_id','question', 'answer', 'score']

  if leave_id_out:
    merged = merged.drop(columns=['question_id'])

  return merged

def normalize_scores(df,leave_min_max_out=True):
  """
  Normalizes scores based on the highest scored answer per question.

    Args:
        df (DataFrame): The given DataFrame.
        leave_min_max_out (bool): Boolean switch for the max_score column.

    Returns:
        DataFrame: the normalized DataFrame.
  """
  # max_score and min_score by question
  df['max_score'] = df.groupby('question')['score'].transform('max')
  df['min_score'] = df.groupby('question')['score'].transform('min')

  # normalize, offset by min to guarantee the value being between 0 and 1
  df['normalized_score'] = (df['score'] - df['min_score']) / (df['max_score'] - df['min_score'])

  if leave_min_max_out:
    df = df.drop(columns=['max_score','score','min_score'])
    df.rename(columns={'normalized_score':'score'},inplace=True)

  return df