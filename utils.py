import pandas as pd
from lxml import etree
import re
from collections import defaultdict, Counter
import praw
import json
import time

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


def custom_tokenizer(text):
  """
  Tokenizes a given string into words, preserving apostrophes.
  
  This tokenizer is a regex-based solution that works by matching
  sequences of alphanumeric characters and apostrophes that are
  bounded by word boundaries. This ensures that words with apostrophes
  are treated as a single token, rather than separate tokens.
  
  :param text: The string to be tokenized.
  :type text: str
  
  :return: A list of words.
  :rtype: list
  """
  return re.findall(r"\b\w[\w']+\b", text.lower())

def count_common_words(question, answer):
  """
  Counts the number of common words between a question and answer.

  Given a question and an answer, this function tokenizes both strings and
  counts the number of occurrences of each common word between them.

  :param question: The question string
  :type question: str
  :param answer: The answer string
  :type answer: str

  :return: A dictionary where the keys are the words and the values are the number of occurrences
  :rtype: dict
  """
  question_words = custom_tokenizer(question)
  answer_words = custom_tokenizer(answer)

  # sets for easier checks
  question_set = set(question_words)
  answer_set = set(answer_words)

  # intersect
  common_words = question_set.intersection(answer_set)

  # count how many there are in both sets
  word_count = Counter()
  word_count.update([word for word in question_words if word in common_words])
  word_count.update([word for word in answer_words if word in common_words])

  return dict(word_count)


def get_cw_feature(df):
  """
  Counts the number of common words between each question and answer in a given
  DataFrame and appends it as a new column named 'common_words'.
  
  :param df: The given DataFrame.
  :type df: pandas.DataFrame
  
  :return: The DataFrame with the 'common_words' column appended.
  :rtype: pandas.DataFrame
  """  
  common_words_list = []
  # iterate through
  for idx, row in df.iterrows():
        question = row['question']
        answer = row['answer']

        # append common words to a list
        common_words = count_common_words(question, answer)
        common_words_list.append(common_words)

  df['common_words'] = common_words_list
  return df

def get_sentences(text):
  """
  Tokenizes a given text into sentences.

  This function takes a given text and uses a regex to break it into sentences.
  The sentences are then returned as a list.

  :param text: The text to be tokenized.
  :type text: str

  :return: A list of sentences.
  :rtype: list
  """
  sentences = []
  regex = r'([A-z][^.!?]*[.!?]*"?)'
  for sentence in re.findall(regex, text):
    sentences.append(sentence)
  return sentences

def get_question_sentences(sentences):
  """
  Finds all sentences in a given list of sentences that end with a question mark.

  This function takes a list of sentences and returns a list of sentences that
  end with a question mark.

  :param sentences: The list of sentences to be searched.
  :type sentences: list

  :return: A list of sentences that end with a question mark.
  :rtype: list
  """
  questions = []
  for sentence in sentences:
    if sentence[-1] == '?':
      questions.append(sentence)
  return questions

def get_posts_from_subreddit(CLIENT_ID,CLIENT_SECRET,USER_AGENT,SUBREDDIT_NAME):

  reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
  )

  subreddit = reddit.subreddit(SUBREDDIT_NAME)

  posts = []
  for submission in subreddit.new(limit=None):
    submission_data = {
        'title': submission.title,
        'selftext': submission.selftext,
        'upvotes': submission.score,
        'created_utc': submission.created_utc,
        'id': submission.id,
        'author': str(submission.author),
        'comments': []
      }

    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        submission_data['comments'].append({
            'author': str(comment.author),
            'comment_body': comment.body,
            'comment_ups': comment.score,
            'created_utc': comment.created_utc
          })

    posts.append(submission_data)

    # add delay to not get locked out
    time.sleep(1)

  with open(f'{SUBREDDIT_NAME}_posts_and_comments.json', 'w') as file:
    json.dump(posts, file, indent=4)

  print(f"Collected {len(posts)} posts from r/{SUBREDDIT_NAME} including comments")

