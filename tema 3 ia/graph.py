import matplotlib.pyplot as plt

'''plt.plot(xb, yb, label = "bleu "+ str(k) +"")
plt.plot(xr, yr, label = "rouge "+ str(k) +"")
plt.xlabel('doc number')
plt.ylim(0, 1)
plt.legend()
plt.show()'''


left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 
# heights of bars
'''height = [0.6569459218116729, 0.6928991716701964, 0.7275347559495742,
          0.6309704310230276, 0.5990211478420028, 0.5910421232921976, 0.6280303155146706,
          0.6209359668714137, 0.624784802906049, 0.5351899255434628, 0.6953635581922839 ]'''

height = [0.5411218507106947 , 0.5335882543319638, 0.5353925316085677,
         0.727058594270946, 0.6531608203464198, 0.6924811383520532, 0.7242713433558859,
          0.7180903765580317, 0.7222199303471122, 0.7604902645304347, 0.6859301023697069 ]
 
# labels for bars
tick_label = ["bayes\nstop words", "bayes\nno stop words", "bayes\nno stop words\nwith lemantization",
              'tf-idf\nwith everything' ,'tf-idf\nno noun', 'tf-idf\nno title similarity', 'tf-idf\nno weight',
              'tf-idf\ncu weight 0.4', 'tf-idf\ncu weight 1.2', 'tf-idf\ncu sent counnt 3', 'tf-idf\ncu sent counnt 5']

plt.ylim(0, 1) 
# plotting a bar chart
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green'])
 
# naming the x-axis
plt.xlabel('sumarization variations')
# naming the y-axis
plt.ylabel('average bleu k = 1')
# plot title
#plt.title('My bar chart!')
 
# function to show the plot
plt.show()