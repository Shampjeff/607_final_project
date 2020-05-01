# This is the main prediction engine for the deployed model in AWS. This file takes inputs from the web app and computes the needed dataframe values (containment and lcs score). It takes the calculated dataframe and passes the numbers to an AWS model endpoint. Once a response has been returned this file post-processes the result for display on the web app

#Currently, the response from the model is not rendered beautifully in html, but this was hard enough for me to get running but it works locally!


import pandas as pd
import numpy as np
import re
import sys
import requests
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

source_a= "in object oriented programming inheritance is a way to form new classes instances of which are called objects using classes that have already been defined the inheritance concept was invented in 1967 for simula  the new classes known as derived classes take over or inherit attributes and behavior of the pre existing classes which are referred to as base classes or ancestor classes  it is intended to help reuse existing code with little or no modification  inheritance provides the support for representation by categorization in computer languages categorization is a powerful mechanism number of information processing crucial to human learning by means of generalization what is known about specific entities is applied to a wider group given a belongs relation can be established and cognitive economy less information needs to be stored about each specific entity only its particularities  inheritance is also sometimes called generalization because the is a relationships represent a hierarchy between classes of objects for instance a fruit is a generalization of apple  orange  mango and many others one can consider fruit to be an abstraction of apple orange etc conversely since apples are fruit i e  an apple is a fruit  apples may naturally inherit all the properties common to all fruit such as being a fleshy container for the seed of a plant  an advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code reducing the complexity of the program inheritance therefore has another view a dual called polymorphism which describes many pieces of code being controlled by shared control code inheritance is typically accomplished either by overriding replacing one or more methods exposed by ancestor or by adding new methods to those exposed by an ancestor  complex inheritance or inheritance used within a design that is not sufficiently mature may lead to the yo yo problem"

source_b= "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents such as the world wide web with the purpose of measuring its relative importance within the set the algorithm may be applied to any collection of entities with reciprocal quotations and references the numerical weight that it assigns to any given element e is also called the pagerank of e and denoted by pr e  the name pagerank is a trademark of google and the pagerank process has been patented u s patent 6 285 999  however the patent is assigned to stanford university and not to google google has exclusive license rights on the patent from stanford university the university received 1 8 million shares in google in exchange for use of the patent the shares were sold in 2005 for 336 million google describes pagerank pagerank relies on the uniquely democratic nature of the web by using its vast link structure as an indicator of an individual page s value in essence google interprets a link from page a to page b as a vote by page a for page b but google looks at more than the sheer volume of votes or links a page receives it also analyzes the page that casts the vote votes cast by pages that are themselves important weigh more heavily and help to make other pages important in other words a pagerank results from a ballot among all the other pages on the world wide web about how important a page is a hyperlink to a page counts as a vote of support the pagerank of a page is defined recursively and depends on the number and pagerank metric of all pages that link to it  incoming links  a page that is linked to by many pages with high pagerank receives a high rank itself if there are no links to a web page there is no support for that page google assigns a numeric weighting from 0 10 for each webpage on the internet this pagerank denotes a site s importance in the eyes of google the pagerank is derived from a theoretical probability value on a logarithmic scale like the richter scale the pagerank of a particular page is roughly based upon the quantity of inbound links as well as the pagerank of the pages providing the links it is known that other factors e g relevance of search words on the page and actual visits to the page reported by the google toolbar also influence the pagerank in order to prevent manipulation spoofing and spamdexing google provides no specific details about how other factors influence pagerank numerous academic papers concerning pagerank have been published since page and brin s original paper in practice the pagerank concept has proven to be vulnerable to manipulation and extensive research has been devoted to identifying falsely inflated pagerank and ways to ignore links from documents with falsely inflated pagerank other link based ranking algorithms for web pages include the hits algorithm invented by jon kleinberg used by teoma and now ask com  the ibm clever project and the trustrank algorithm"

source_c= "vector space model or term vector model is an algebraic model for representing text documents and any objects in general as vectors of identifiers such as for example index terms it is used in information filtering information retrieval indexing and relevancy rankings its first use was in the smart information retrieval system a document is represented as a vector each dimension corresponds to a separate term if a term occurs in the document its value in the vector is non zero several different ways of computing these values also known as term weights have been developed one of the best known schemes is tf idf weighting see the example below  the definition of term depends on the application typically terms are single words keywords or longer phrases if the words are chosen to be the terms the dimensionality of the vector is the number of words in the vocabulary the number of distinct words occurring in the corpus  the vector space model has the following limitations 1 long documents are poorly represented because they have poor similarity values a small scalar product and a large dimensionality 2 search keywords must precisely match document terms word substrings might result in a false positive match 3 semantic sensitivity documents with similar context but different term vocabulary won t be associated resulting in a false negative match 4 the order in which the terms appear in the document is lost in the vector space representation"

source_d= "in probability theory bayes theorem often called bayes law after rev thomas bayes relates the conditional and marginal probabilities of two random events it is often used to compute posterior probabilities given observations for example a patient may be observed to have certain symptoms bayes theorem can be used to compute the probability that a proposed diagnosis is correct given that observation  see example 2 as a formal theorem bayes theorem is valid in all common interpretations of probability however it plays a central role in the debate around the foundations of statistics frequentist and bayesian interpretations disagree about the ways in which probabilities should be assigned in applications frequentists assign probabilities to random events according to their frequencies of occurrence or to subsets of populations as proportions of the whole while bayesians describe probabilities in terms of beliefs and degrees of uncertainty the articles on bayesian probability and frequentist probability discuss these debates in greater detail bayes theorem relates the conditional and marginal probabilities of events a and b where b has a non vanishing probability p a b frac p b  a  p a  p b  each term in bayes theorem has a conventional name  p a is the prior probability or marginal probability of a it is prior in the sense that it does not take into account any information about b  p a b is the conditional probability of a given b it is also called the posterior probability because it is derived from or depends upon the specified value of b  p b a is the conditional probability of b given a  p b is the prior or marginal probability of b and acts as a normalizing constant intuitively bayes theorem in this form describes the way in which one s beliefs about observing a are updated by having observed"

source_e= "in mathematics and computer science dynamic programming is a method of solving problems that exhibit the properties of overlapping subproblems and optimal substructure described below  the method takes much less time than naive methods the term was originally used in the 1940s by richard bellman to describe the process of solving problems where one needs to find the best decisions one after another by 1953 he had refined this to the modern meaning the field was founded as a systems analysis and engineering topic that is recognized by the ieee bellman s contribution is remembered in the name of the bellman equation a central result of dynamic programming which restates an optimization problem in recursive form the word programming in dynamic programming has no particular connection to computer programming at all and instead comes from the term mathematical programming  a synonym for optimization thus the program is the optimal plan for action that is produced for instance a finalized schedule of events at an exhibition is sometimes called a program programming in this sense means finding an acceptable plan of action an algorithm optimal substructure means that optimal solutions of subproblems can be used to find the optimal solutions of the overall problem for example the shortest path to a goal from a vertex in a graph can be found by first computing the shortest path to the goal from all adjacent vertices and then using this to pick the best overall path as shown in figure 1 in general we can solve a problem with optimal substructure using a three step process 1 break the problem into smaller subproblems 2 solve these problems optimally using this three step process recursively 3 use these optimal solutions to construct an optimal solution for the original problem the subproblems are themselves solved by dividing them into sub subproblems and so on until we reach some simple case that is solvable in constant time figure 2 the subproblem graph for the fibonacci sequence that it is not a tree but a dag indicates overlapping subproblems to say that a problem has overlapping subproblems is to say that the same subproblems are used to solve many different larger problems for example in the fibonacci sequence f3  f1  f2 and f4  f2  f3  computing each number involves computing f2 because both f3 and f4 are needed to compute f5 a naive approach to computing f5 may end up computing f2 twice or more this applies whenever overlapping subproblems are present a naive approach may waste time recomputing optimal solutions to subproblems it has already solved in order to avoid this we instead save the solutions to problems we have already solved then if we need to solve the same problem later we can retrieve and reuse our already computed solution this approach is called memoization not memorization although this term also fits  if we are sure we won t need a particular solution anymore we can throw it away to save space in some cases we can even compute the solutions to subproblems we know that we ll need in advance"


input_text = sys.argv[1]
input_text_2 = sys.argv[2]
input_text_3 = sys.argv[3]
input_text_4 = sys.argv[4]
input_text_5 = sys.argv[5]
source_id = sys.argv[6]
source_id_2 = sys.argv[7]
source_id_3 = sys.argv[8]
source_id_4 = sys.argv[9]
source_id_5 = sys.argv[10]



def process_file(file):
    all_text = file.lower()
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    
    return all_text

def make_count_vector(corpus, ngrams):
    
    try: 
        CountVectorizer()
    except IOError:
        print('Import CountVectorizer from sklean.feature_extraction.text')
    count_vec = CountVectorizer(analyzer='word', ngram_range=(ngrams,ngrams))
    return count_vec.fit_transform(corpus)

def calculate_containment(df, n): 
   
    source = df["source"]
    source_idx = source.index[0]

    answer = df["answer"]
    answer_idx = answer.index[0]    
    
    count_vec = make_count_vector([answer[answer_idx], source[source_idx]], n)
    intersect_sum = np.minimum(count_vec[0].toarray(), count_vec[1].toarray()).sum()
      
    answer_sum = count_vec[0].toarray().sum()
    containment = intersect_sum/answer_sum
    
    return containment

def lcs_norm_word(answer_text, source_text):
 
    ans = answer_text.split()
    src = source_text.split()
    mat = np.zeros((len(ans)+1, len(src)+1))
    
    for i in range(len(ans)):
        for j in range(len(src)):
            if ans[i] == src[j]:
                mat[i+1,j+1] = mat[i,j]+1
            else: 
                mat[i+1,j+1] = max(mat[i,j+1], mat[i+1,j])
                
    lcs_score = mat[-1,-1] / len(ans)
    return lcs_score


def create_containment_features(df, n, column_name=None):
    
    containment_values = []
    
    if(column_name==None):
        column_name = 'c_'+str(n) 
    
    c = calculate_containment(df, n) #file name arg
    containment_values.append(c)
    
    return containment_values


def create_lcs_features(df, column_name='lcs_word'):
    
    lcs_values = []
    answer_text=df["answer"][0]
    source_text=df["source"][0]
            # Calculate lcs
    lcs = lcs_norm_word(answer_text, source_text)
    lcs_values.append(lcs)
       
    return lcs_values



#def lambda_handler(event, context):
    
source_dict ={"source_a":source_a, "source_b":source_b, "source_c":source_c,
                "source_d":source_d, "source_e":source_e}
ngram_range = range(1,9)

# Tag the source file possible using the POST function with API Gateway

def make_data_web_app(input_text, source_id, sources=source_dict, ngram_range=ngram_range):
    complete_df= pd.DataFrame(columns=['source','answer'])
    complete_df= complete_df.append({"source":sources[source_id],
                                "answer":process_file(input_text)},
                   ignore_index=True)

    all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

    features_list = []
    # Calculate features for containment for ngrams in range
    i=0
    for n in ngram_range:
        column_name = 'c_'+str(n)
        features_list.append(column_name)
        all_features[i]=np.squeeze(create_containment_features(complete_df, n))
        i+=1

    # Calculate features for LCS_Norm Words 
    features_list.append('lcs_word')
    all_features[i]= np.squeeze(create_lcs_features(complete_df))

    # create a features dataframe
    features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)
    selected_features = ['c_1', 'c_2','c_8', 'lcs_word']

    features_df= features_df[selected_features]
    #return(features_df.to_numpy().tolist()[0])
    return features_df

out=make_data_web_app(input_text, source_id)
out_2=make_data_web_app(input_text_2, source_id_2)
out_3=make_data_web_app(input_text_3, source_id_3)
out_4=make_data_web_app(input_text_4, source_id_4)
out_5=make_data_web_app(input_text_5, source_id_5)


out= pd.concat([out, out_2, out_3, out_4, out_5])


import io
import boto3
import json

from io import StringIO

test_file = io.StringIO()
out.to_csv(test_file,header = None, index = None)


endpoint_name= 'gbm-endpoint-607'
client = boto3.Session().client('sagemaker-runtime', 
                        region_name='us-east-2')
response = client.invoke_endpoint(
    EndpointName= endpoint_name,
    Body= test_file.getvalue(),
    ContentType = 'text/csv')
result = json.loads(response['Body'].read().decode())

j=0
for i in result:
    j+=1
    if i ==0:
        out_text= f"Question {j} answer: authentic   "
        sys.stdout.write(out_text)
    else:
        out_text =f"Question {j} answer: not authentic   "
        sys.stdout.write(out_text)



