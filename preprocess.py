import sys
import json
import re


class Preprocessor:
    def __init__(self):
        self.html_re = r'</?\w+[^>]*>'
        self.reg = re.compile(self.html_re)
    
    def rm_html(self, content):
        content = self.reg.sub('',content)
        return content


class Preprocessor_En(Preprocessor):
    def __init__(self):
        super(Preprocessor_En,self).__init__()
        
        self.sen_split_type = 'nltk'
        self.word_tokenizer_type = None
        self.word_tokenizer = None 

    def tokenize(self, content, model='zzy'):
        if model == 'nltk':
            import nltk
            res = nltk.word_tokenize(content)
        if model == 'spacy':
            
            if self.word_tokenizer_type != 'spacy':
                import spacy
                self.word_tokenizer_type = 'spacy'    
                self.word_tokenizer = spacy.load('en_core_web_sm')
            doc = self.word_tokenizer(content)
            res = [t.text for t in doc]
        if model == 'stanfordnlp':
            if self.word_tokenizer_type != 'stanfordnlp':
                import stanfordnlp
                nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
                self.word_tokenizer_type = 'stanfordnlp'
                self.word_tokenizer = nlp
            doc = self.word_tokenizer(content)
            res = []
            for sentence in doc.sentences:
                res.extend([token.text for token in sentence.tokens])
        if model == 'zzy':
            if self.word_tokenizer_type != 'zzy':
                import spacy
                self.word_tokenizer_type = 'zzy'    
                self.word_tokenizer = spacy.load('en_core_web_sm')
            doc = self.word_tokenizer(content)
            cur = [t.text for t in doc]
            pat = "(\.|\?|-|:|&|/|')"
            res = []
            for word in cur:
                for span in re.split(pat,word):
                    if span != '' and span!= ' ' :
                        if span in {'.','-'} and len(res)>0 and span in res[-1]:
                            res[-1] = res[-1]+span
                        else:
                            res.append(span)
                        
        return ' '.join(res)



if __name__ == '__main__':
    html_en = "<body><div><img src=\"http://image1.hipu.com/image.php?type=_640x390&amp;url=4WeoLx_0MtyePAH00\" alt=\"http://image1.hipu.com/image.php?url=4WeoLx_0MtyePAH00\" width=\"320\" height=\"195\"><p>A little boy took a menstrual cup to school to show off during show and tell after spending several days playing with it while his mother was out of town.</p><p>His embarrassed dad shared the story on Reddit this week, explaining that he had no idea what the little plastic cup his son was playing with actually was.</p><p>It wasn't until the little boy's teacher asked to speak to him after school that he learned his son's favorite new toy was actually used for collecting menstrual blood \u2014 after his poor teacher had to explain in detail how it worked.</p><figure><img src=\"http://image1.hipu.com/image.php?type=_640x390&amp;url=4du2lx_0MtyePAH00\" alt=\"http://image1.hipu.com/image.php?url=4du2lx_0MtyePAH00\" width=\"320\" height=\"195\"></figure><p>The anonymous dad broke down what happened in the 'Today I F***ed Up' forum on Reddit. </p><p>He wrote that his wife is spending a couple of weeks working out of town, so he's home alone with their son Ben.</p><p>At one point over the weekend, he noticed Ben was playing with 'this little silicone cup that kinda looked like a tulip.'</p><p>The dad had obviously never seen a menstrual cup before, as they're less commonly used than tampons and pads. He assumed it was a toy. </p><p>'I asked him what it went to and what it did and he proceeded to show me it\u2019s versatility,' he wrote. 'Over the next few days it helped the Paw Patrol save the town, it was a treasure chest holding tiny pebbles guarded by pirates, a force field protecting a space ship. </p><p>'It came with us to the park, grocery shopping, and even out to dinner one night. I loved that it had its own little satchel and assumed it just went to a play set,' he explained. </p><p>Then, on Wednesday morning, Ben had to bring something in for show and tell. </p><p>'So my son grabs his little silicone cup and off to school he goes,' the dad wrote. </p><p>But when he came back to pick him up after school, the teacher asked to have a word \u2014 a conversation which he recounted in his post. </p><p>'Ben\u2019s show and tell was...interesting,' the teacher said.</p><p>'Yea! It\u2019s cool right? We\u2019ve been playing with that thing for days,' he replied.</p><p>'Uh, Mr. Scott, do you know what that is? ...That is a, uh, menstrual cup,' she told him.</p><p>The dad said this was confusing to him, and the teacher seemed to notice he was having a hard time making the connection. </p><p>'It\u2019s um, used to collect menstrual blood...' she went on as he continued to look confused. 'Uh, goes inside, and uh... collects blood.'</p><p>'It just...stays in there?' he asked, and she nodded. 'Are you sure? I don\u2019t think that would, uh, fit....too, uh...comfortably...there.'</p><p>Oh it folds in half then springs open inside,' she told him, leaving them both uncomfortable.</p><p>'Alright then,' he answered. 'So where do I get a replacement because my wife will probably not be too pleased when she returns home and will not want to continue using this one.'</p><p>The dad said that his wife 'laughed hysterically' when he told her, and his son is still using the object as a toy. </p><p>Later, the confused dad clarified that he is aware of other menstruation products, but knows tampons are much smaller than cups, so he was confused about the fit. </p><p>'I\u2019d like to try to upload a picture of the cup in the back of a tractor with a chicken riding in it but am waiting on my wife\u2019s approval,' he added.</p><p>Other Redditors found the story quite funny, and some shared their own similar stories.</p><p>'This is awesome! When we were little, my brother and found my dad's condoms and we spent an afternoon on the front lawn playing with them. \"Worst water balloons ever. And why are they individually wrapped?\"' one wrote.</p><p>'My son found mine (before I used them thank god) and thought they were toys too,' wrote one woman. 'I had just opened them so I left them in my room bc I was reading the directions and stuff. He proceeded to put them on his face bc they looked like \u201cfunny cone eyes\u201d ... I still have the pictures.'</p><p>'Reminded me of a boy in my son's first grade class,' began another. 'The little entrepreneur was selling telescopes for a nickel each, and sold out. Sadly, they were his mother's used tampon applicators.'</p></div></body>"
    
    processor = Preprocessor_En()
    ori_content = processor.rm_html(html_en)
    print(ori_content)
    
    content = processor.tokenize(ori_content, model = 'nltk')
    print(content)
    content = processor.tokenize(ori_content, model = 'spacy')
    print(content)
    content = processor.tokenize(ori_content, model = 'stanfordnlp')
    print(content)
    content = processor.tokenize(ori_content, model = 'zzy')
    print(content)
