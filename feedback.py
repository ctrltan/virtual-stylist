import matplotlib.pyplot as plt
import cv2
import os
import sys

class FeedbackMachine:

    def __init__(self, outfit, inspirations):
        self.outfit = outfit
        self.overall_score = 0
        self.inspirations = inspirations
        self.feedback = {}
        cont = input("Do you wish to continue? ")
        if cont == 'no':
            sys.exit()
        self.display_outfit()
        self.collect_overall_score()
        self.collect_feedback()
        plt.close()

    def collect_overall_score(self):
        try:
            self.overall_score = int(input("Please score the outfit out of 10: "))
        except:
            print("Invalid")
            return self.collect_overall_score()

        if self.overall_score < 0 or self.overall_score > 10:
            print("Out of range")
            return self.collect_overall_score()

    def get_overall_score(self):
        return self.overall_score

    def collect_feedback(self):
        inspiration_keys = set(self.inspirations)

        all_rankings = []
        for i in range(len(self.outfit)):
            try:
                item_rank = int(input(f"Please score this {self.outfit[i]['Label'][:-1]} out of 10: "))
            except:
                print("Invalid")
                return self.collect_feedback()
            
            if item_rank < 0 or item_rank > 10:
                print("Out of range")
                return self.collect_feedback()
            
            all_rankings.append((self.inspirations[i], item_rank))


        for key in inspiration_keys:
            inspiration_ranks = []
            for (inspiration, rank) in all_rankings:
                if key == inspiration:
                    inspiration_ranks.append(rank)
            self.feedback[key] = inspiration_ranks

    def get_feedback(self):
        return self.feedback
    

    def display_outfit(self):
        labels = [item['Label'] for item in self.outfit]
        label_path_pairs = [(item['Label'], item['Path']) for item in self.outfit]

        display_layout = {'tops': (0, 0),
                          'skirts': (1, 0),
                          'long_skirts': (1, 0),
                          'trousers': (1, 0),
                          'jackets':(0, 1)
                          }
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        for label, path in label_path_pairs:
            outfit_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            outfit_img = cv2.cvtColor(outfit_img, cv2.COLOR_BGRA2RGBA)

            row, col = display_layout[label]

            axes[row, col].imshow(outfit_img)
            axes[row, col].set_title(label[:-1])
        
        for ax in axes.flatten():
            ax.axis('off')

        
        plt.tight_layout()
        plt.show(block=False)

        
