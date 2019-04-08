import sys
from scapy.all import *
from scapy.all import IP,TCP
import getmac

import getmac
import collections
#from scapy.all import rdpcap
import numpy as np
#import scapy.all as sa
from scapy.all import *
import tensorflow as tf
#import scipy
#/////////////////////////////////////////////////////////////////////////
#my functions
import math
import operator
import collections
import statistics
#import itertools
import copy
import time

#input: myarray-> array you want to append zeros
#       size-> outputsize of the array
#output: array
def appendzeros(myarray,size):
    for i in range(0,size):
        if i not in myarray.keys():
            myarray[i] = 0
        else:
            myarray[i] = myarray[i]
    return myarray

#input: dic fromat
#output: list format
def dic2list_nolz(myarray):
    temp=[]
    for key, value in dict.items(myarray):
        if value != 0:
            #///////////normalizing equation
            temp.append(math.log(key**value,20))
        else:
            temp.append(0)
    return temp

#input: array-> array that you want to shrink
#       size-> shrink ratio
#output:shrinked array
def shrink(array,size):
    temp1=[]
    x=0
    for i in range(0,len(array),size):
        temp1.append(sum(array[x:i])/size)
        #temp1.append(i)
        x=i
    return temp1

#input: array-> array that you want to find maximum values
#       size-> number of maximum points
#output:maximum points array
def maxpoint(array,numpoints):
    mymax=[]
    bukket = dict()
    for i in range(1,numpoints):
        #mymax.append(max(array))
        index, value = max(enumerate(array), key=operator.itemgetter(1))
        #mymax.append(value)
        #array[index]=0
        bukket[index]=value
        array[index]=0
    bukket = collections.OrderedDict(sorted(bukket.items()))
    for key, value in dict.items(bukket):
        mymax.append(value)

    if len(mymax) != 100:
        for item in range(len(mymax),100):
            mymax.append(0)
    return mymax

#input: numarray-> array that you wants to check area
#       range_start-> calculation starting point
#       range_end-> calculation end point
#output:area
def range_area(range_start,range_end,numarray):
    total=0
    for i in range(range_start,range_end):
        total=numarray[i]+total
    return total

#input: numarray-> array that has the checking array index
#       oplist-> number of index that you want check
#output:maximum point
def op_max_checker(oplist,numarray):
    if numarray[oplist] > numarray[oplist+1] and numarray[oplist] > numarray[oplist-1] and numarray[oplist]/2 > (numarray[oplist-1]+numarray[oplist+1]):
        return numarray[oplist]
    else:
        return 0

#input: numarray-> array that you want to calculate std
#       range_start-> calculation starting point
#       range_end-> calculation end point
#output:std of the given list range
def range_std(range_start,range_end,numarray):
    std_list=[]
    std_value=0;
    std_list = copy.copy(numarray[range_start:range_end])
    return statistics.stdev(std_list)


def range_nzlength(range_start,range_end,numarray,fs):
    x=0
    y=1
    temp=[]
    temp.append(0)
    for i in range(range_start,range_end):
        if numarray[i]>0:
            x=x+1
        elif (numarray[i] == 0 and x>0) or i==range_end-1:
            temp.append(x)
            y=y+1
            x=0
        else:
            continue

    if max(temp)>0:
        return max(temp)
    else:
        return 0

def range_max_index(range_start,range_end,numarray):
    y = copy.copy(numarray[range_start:range_end])
    index, value = max(enumerate(y), key=operator.itemgetter(1))
    return index, value

#end of my functions///////////////////////////////////////

#Youtube features ///////////////////////////////////////////
def youtube_f1 (numarray):
    # x=mf.op_max_checker(5434,numarray)
    # if x>1:
    #     return numarray[5434]
    # else:
    #     return 0
    return numarray[5434] if numarray[5434]>0 else 0

def youtube_f2 (numarray):
    x=numarray[3521]+numarray[3522]+numarray[3523]+numarray[3516]
    if x>2.1:
        return x
    else:
        return 0

def youtube_f3 (numarray):
    x=numarray[2849]+numarray[2850]+numarray[2851]+numarray[4571]
    if x>0:
        return x
    else:
        return 0

def youtube_f4 (numarray):
    x=numarray[2513]+numarray[2514]+numarray[2515]+numarray[2516]+numarray[2517]
    if x>1:
        return math.log(x,5)
    else:
        return 0

def youtube_f5 (numarray):
    x=op_max_checker(4100,numarray)
    if x>5:
        return numarray[4100]
    else:
        return 0

#added 3660 and 3664 because of testbed
def youtube_f6 (numarray):
    x = numarray[3661] + numarray[3662] + numarray[3663]
    if x>3:
        return math.log(x,5)
    else:
        return 0

def youtube_f7 (numarray):
    x=0
    x=range_area(2485,2493,numarray)
    #x=numarray[2485]+numarray[2493]
    if x>3:
        return math.log(x,10)
    else:
        return 0
#following cunctions were added to handle testbed data ===========
def youtube_f8(numarray):
    x = numarray[4113] + numarray[4112]
    return x if x>0 else 0

def youtube_f9(nummarray):
    x = nummarray[3418] + nummarray[3204] + nummarray[3106]
    return x if x>0 else 0

def youtube_f10(numarray):
    x = range_area(5100,5650,numarray)
    return x if x>0 else 0

def youtube_f11 (numarray):
    x = numarray[3863] + numarray[3859]
    return x if x>0 else 0
#End of youtube features ////////////////////////////////////

#Facebook features/////////////////////////////////////////
def facebook_f1 (numarray):
    x=op_max_checker(3903,numarray)
    if x>5:
        return math.log(x,10)
    else:
        return 0

def facebook_f2 (numarray):
    if (numarray[2847])>0:
        return math.log(numarray[2847],5)
    else:
        return 0

def facebook_f3 (numarray):
    if (numarray[2626])>0:
        x=math.log(numarray[2626],5)
        return x
    else:
        return 0

def facebook_f4 (numarray):
    if numarray[2535]>0:
        x=math.log(numarray[2535],10)
        return x
    else:
        return 0

def facebook_f5 (numarray):
    x=0
    x=range_area(4193,4203,numarray)
    if x>0:
        #math.log(x,10)
        return x/10
    else:
        return 0

def facebook_f6 (numarray):
    if (numarray[5464]+numarray[5494]+numarray[5506])>0:
        x=math.log(numarray[5464]+numarray[5494]+numarray[5506],5)
        return x
    else:
        return 0

#following cunctions were added to handle testbed data ===========
def facebook_f7(numarray):
    x = numarray[3899] + numarray[3900]
    return x if x>0 else 0

def facebook_f8(numarray):
    x = range_area(4302,4334,numarray)
    return x if x>2 else 0

def facebook_f9(numarray):
    x = numarray[3904] + numarray[3905] + numarray[3906]
    return x if x>0 else 0
#End of facebook features//////////////////////////////////

#Whatsapp features //////////////////////////////////////////
def whatsapp_f1f2f3 (numarray):
    x=0
    #x=mf.range_nzlength(3770,3832,numarray,6)
    #x=[x+1 for i in range(3781,3860) if numarray[i]>0 else continue]
    for i in range(3780,3859):
        if numarray[i]>2:
            x+=1
        else:
            continue

    #index, value = mf.range_max_index(3770,3832,numarray)
    if x>4 and (numarray[4194]+numarray[4196]+numarray[4205])<10 and numarray[3861]+numarray[3859] == 0:
        wstd=range_std(3780,3859,numarray)
        warea=range_area(3780,3859,numarray)
        return x,wstd,warea/10
    else:
        return 0,0,0

def whatsapp_f4 (numarray):
    #x=mf.op_max_checker(3835,numarray)
    if numarray[3835]>0:
        return numarray[3835]
    else:
        return 0


def whatsapp_f5 (numarray):
    #if numarray[4165] + numarray[4137]>4:
    if numarray[4164]>4:
        return numarray[4164] + numarray[4136]
    else:
        return 0

    #return numarray[4165] + numarray[4137] if umarray[4165] + numarray[4137]>0 else 0

def whatsapp_f6 (numarray):
    x=numarray[4068]+numarray[4069]+numarray[4070]
    if x>0:
        return x
    else:
        return 0

def whatsapp_f7 (numarray):
    x=numarray[4042]+numarray[4048]
    if x>0:
        return x
    else:
        return 0

def whatsapp_f8 (numarray):
    x=numarray[3955]+numarray[3957]
    if x>0:
        return x
    else:
        return 0

def whatsapp_f9 (numarray):
    x=numarray[3931]+numarray[3929]+numarray[3927]
    if x>5:
        return x*10
    else:
        return 0

def whatsapp_f10f11f12 (numarray):
    x=range_nzlength(4109,4177,numarray,5)
    index, value = range_max_index(4079,4159,numarray)

    if x>4 and (index+4079)>4119 and numarray[2535]+numarray[4114]==0:
        sstd=range_std(4079,4159,numarray)
        sarea=range_area(4079,4159,numarray)
        return x,sstd,sarea
    else:
        return 0,0,0
#End of whatsapp features////////////////////////////////////

#Skype features////////////////////////////////////////////
def skype_f1f2f3 (numarray):
    x=range_nzlength(4090,4150,numarray,6)
    index, value = range_max_index(4090,4150,numarray)

    if x>6 and value>14:
        sstd=range_std(4080,4160,numarray)
        sarea=range_area(4080,4160,numarray)
        return x,sstd,math.log(sarea,10)
    else:
        return 0,0,0

def skype_f4f5f6 (numarray):
    x=range_nzlength(3837,3910,numarray,8)
    index, value = range_max_index(3837,3910,numarray)
    if x>8 and 15<value<85:
        sstd=range_std(3837,3910,numarray)
        sarea=range_area(3837,3910,numarray)
        return x,sstd,math.log(sarea,10)
    else:
        return 0,0,0

def skype_f7 (numarray):
    if (numarray[3861])>0:
        #x=math.log(numarray[3861],5)
        x=numarray[3861]
        return x
    else:
        return 0

def skype_f8 (numarray):
    if (numarray[3885])>0:
        x=math.log(numarray[3885],5)
        return x
    else:
        return 0

def skype_f9 (numarray):
    return math.log(numarray[4114],5) if numarray[4114]>1 else 0


def skype_f10 (numarray):
    x=numarray[3888]+numarray[3861]+numarray[3859]
    if x>15:
        return x
    else:
        return 0
#End of skype features////////////////////////////////////

#Skype Video features//////////////////////////////////////////////
def skypeVDOC_f1f2f3 (numarray):
    x=0
    for i in range(2800,3400):
        if numarray[i]>9:
            x+=1
        else:
            continue

    svdoc_sd=range_std(2800,3400,numarray)
    svdoc_area=range_area(2800,3400,numarray)
    return x,svdoc_sd,math.log(svdoc_area,10) if svdoc_area>0 else 0

def skypeVDOC_f4f5f6 (numarray):
    x=0
    for i in range(4500,5100):
        if numarray[i]>7:
            x+=1
        else:
            continue

    svdoc_sd=range_std(4500,5100,numarray)
    svdoc_area=range_area(4500,5100,numarray)
    #svdoc_area_log=math.log(svdoc_area,10)
    return x,svdoc_sd,math.log(svdoc_area,10) if svdoc_area>0 else 0
#End of Skype vedio features/////////////////////////////////


while True:
    starttime=time.time()
    tf.reset_default_graph()
    #///////////////////////////////////////////////////////////////////////////
    #pkts_list=sa.sniff(timeout=5)
    pkts_list=sniff(iface=conf.iface,timeout=8)

    sent_pktsl=[]
    received_pktsl=[]
    time_p=8

    features=[]
    #'70:71:bc:71:0f:09' ----> desktop
    #'f8:34:41:97:52:bb'  ----> Laptop
    #64:5a:04:7c:12:c3 ---> dumindu
    #//////////////////filtering based on MAC add. and time
    #print(pkts_list)
    my_MAC=getmac.get_mac_address()
    #print(my_MAC)
    for i in pkts_list:
        MyPacket = i
        if (i.src == my_MAC):
            sent_pktsl.append(len(i))
        elif (i.dst == my_MAC):
            received_pktsl.append(len(i))
        else:
            continue

    #////////////////reading receiving packets and reversing the order
    #print(received_pktsl)
    test1=sorted(received_pktsl,reverse=True)
    #test1.sort(key=int)
    #print(test1)
    #test1=sorted(received_pktsl,key=int)

    bukket1 = dict()
    for elem in test1:
        if elem not in bukket1.keys():
            bukket1[elem] = 1
        else:
            bukket1[elem] += 1

    #////////////////reading the sent packets
    test2=sorted(sent_pktsl, key=int)
    #nop=Counter(test2)

    bukket2 = dict()
    for elem in test2:
        if elem not in bukket2.keys():
            bukket2[elem] = 1
        else:
            bukket2[elem] += 1

    #//////////// appending zeros to generated receiving data array to make it a
    # fixed sized array for every data sample(1mint packets)
    bukket1 = appendzeros(bukket1,4000)
    #///////////arranging the appended array in revers order based on dict's key value
    bukket1 = collections.OrderedDict(sorted(bukket1.items(),reverse=True))

    listbukket1=[]
    listbukket1 = dic2list_nolz(bukket1)
    #//////////// appending zeros to generated sent data array to make it  a fixed
    # sized array for every data sample(1mint packets)
    bukket2 = appendzeros(bukket2,4000)
    #///////////arranging the appended array in revers order based on dict's key value
    bukket2 = collections.OrderedDict(sorted(bukket2.items()))

    listbukket2=[]
    listbukket2 = dic2list_nolz(bukket2)

    #//////////Concatenating two matrices
    listbukket1=listbukket1+listbukket2

    #YouTube Vedio===========================================================
    features.append(youtube_f1(listbukket1))
    features.append(youtube_f2(listbukket1))
    features.append(youtube_f3(listbukket1))
    features.append(youtube_f4(listbukket1))
    features.append(youtube_f5(listbukket1))
    features.append(youtube_f6(listbukket1))
    features.append(youtube_f7(listbukket1))
    features.append(youtube_f8(listbukket1))
    features.append(youtube_f9(listbukket1))
    features.append(youtube_f10(listbukket1))
    features.append(youtube_f11(listbukket1))
    # features.append(features[0] + features[1] + features[2] + features[3] + features[4] + features[5]
    #                 + features[6] + features[7] + features[8] + features[9] + features[10])
    #Facebook Vedio ===========================================================
    features.append(facebook_f1(listbukket1))
    features.append(facebook_f2(listbukket1))
    features.append(facebook_f3(listbukket1))
    features.append(facebook_f4(listbukket1))
    features.append(facebook_f5(listbukket1))
    features.append(facebook_f6(listbukket1))
    features.append(facebook_f7(listbukket1))
    features.append(facebook_f8(listbukket1))
    features.append(facebook_f9(listbukket1))
    # features.append(features[12] + features[13] + features[14] + features[15] + features[16] + features[17]
    #                 + features[18] + features[19] + features[20])
    #WhatsApp call===========================================================
    x,y,z = whatsapp_f1f2f3(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    features.append(whatsapp_f4(listbukket1))
    features.append(whatsapp_f5(listbukket1))
    features.append(whatsapp_f6(listbukket1))
    features.append(whatsapp_f7(listbukket1))
    features.append(whatsapp_f8(listbukket1))
    features.append(whatsapp_f9(listbukket1))
    x,y,z=whatsapp_f10f11f12(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    # features.append(features[22] + features[23] + features[24] + features[25] + features[26] + features[27]
    #                 + features[28] + features[29] + features[30] + features[31] + features[32] + features[33])
    #Skype call===========================================================
    x,y,z = skype_f1f2f3(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    x,y,z = skype_f4f5f6(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    features.append(skype_f7(listbukket1))
    features.append(skype_f8(listbukket1))
    features.append(skype_f9(listbukket1))
    features.append(skype_f10(listbukket1))
    # features.append(features[35] + features[36] + features[37] + features[38] + features[39] + features[40]
    #                 + features[41] + features[42] + features[43] + features[44])
    #Skype Vedio call===========================================================
    x,y,z=skypeVDOC_f1f2f3(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    x,y,z=skypeVDOC_f4f5f6(listbukket1)
    features.append(x)
    features.append(y)
    features.append(z)
    #features.append(features[46] + features[47] + features[48] + features[49] + features[50] + features[51])
    #//////////shrinking the concatienated llist
    #new=[]
    #new=mf.shrink(listbukket1,4)
    #//////////getting the maximum values of it as features
    #new=mf.maxpoint(new,100)

    #for item3 in new:
        #trainingd.write("%s\n" % item3)

    #////////////////////////////////////////
    features=np.array(features)
    my_list=features.reshape(1,-1)
    inputs=tf.placeholder('float') # inputdat
    #label=tf.placeholder(tf.float32,shape=nn_target_data[1],name='labels') # lables of the x
    label=tf.placeholder('float') # lables of the x

    # input layer
    hid1_size = 48
    w1 = tf.Variable(tf.random_normal([hid1_size, my_list.shape[1]],stddev= 0.1, seed=1), name='w1')
    b1 = tf.Variable(tf.random_normal(shape=(hid1_size, 1)), name='b1')
    y1 = tf.nn.sigmoid(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1))

    # 2nd layer
    hid2_size = 20
    w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size],stddev= 0.1, seed=1), name='w2')
    b2 = tf.Variable(tf.random_normal(shape=(hid2_size, 1)), name='b2')
    y2 = tf.nn.sigmoid(tf.add(tf.matmul(w2, y1), b2))

    # # 3rd layer
    # hid3_size = 5
    # w3 = tf.Variable(tf.random_normal([hid3_size, hid2_size],stddev= 0.1, seed=1), name='w2')
    # b3 = tf.Variable(tf.random_normal(shape=(hid3_size, 1)), name='b2')
    # y3 = tf.nn.sigmoid(tf.add(tf.matmul(w3, y2), b3))

    # Output layer
    wo = tf.Variable(tf.random_normal([5, hid2_size], stddev= 0.1, seed=1), name='wo')
    bo = tf.Variable(tf.random_normal([5, 1]), name='bo')
    prediction = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

    # Loss function and optimizer

    # Create operation which will initialize all variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    model_path="save_net.ckpt"
    detection_graph = tf.Graph()

    with tf.Session() as sess:
        sess.run(init)
        #saver.restore(sess, "save_net.ckpt")
        loader = tf.train.import_meta_graph(model_path+'.meta')
        loader.restore(sess, model_path)
        # For each epoch, we go through all the samples we have.
        for i in range(my_list.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            test_predict = sess.run(prediction, feed_dict={inputs: my_list[i, None]}).squeeze()
            # for i in range(X_test.shape[0]):
            #     df_test.loc[i, 'Survived'] = sess.run(pred_label, feed_dict={inputs: X_test[i, None]}).squeeze()

            # save_path = saver.save(sess, "C:/Users/Rimas/Desktop/MSc/My project/5sec_NN_my/nn_save/save_net.ckpt")

            #rawdataR = open("C:\\Users\\Rimas\\Desktop\\MSc\\IEEE\\attempt 2\\images\\predictions.csv", "w")
            pred = tf.argmax(tf.nn.softmax(prediction),1)
            predicted_class=int(pred.eval({inputs:my_list}))

        MyPacket.load = Raw(None)

        if predicted_class == 0:
            predicted_class="YouTube,"
            MyPacket = MyPacket/Raw(load="Video")
        elif predicted_class== 1:
            predicted_class="FaceBook,"
            MyPacket = MyPacket/Raw(load="Video")
        elif predicted_class== 2:
            predicted_class="WhatsApp,"
            MyPacket = MyPacket/Raw(load="WhatsAppVoice")
        elif predicted_class== 3:
            predicted_class="SkypeVoiceCall,"
            MyPacket = MyPacket/Raw(load="SkypeVoice")
        elif predicted_class== 4:
            predicted_class="SkypeVideoCall,"
            MyPacket = MyPacket/Raw(load="SkypeVideo")
        else:
            predicted_class="DO_NOT_KNOW"

        with open("history.txt", "a") as output:
            output.write("%s\n" % predicted_class)
        with open("lable.txt", "w") as output:
            output.write("%s\n" % predicted_class)
        # print_predict=tf.cast(prediction,'float')
        # new=list(print_predict.eval({inputs:my_list}))
        # with open("C:\\Users\\Rimas\\Desktop\\predictions\\MLPpredictionsLoaded.csv", "a") as output:
        #     writer=csv.writer(output,lineterminator='\n')
        #     writer.writerows(new)

    #print(i.payload)
    #MyPacket.load = Raw(None)
    #MyPacket = MyPacket/Raw(load="Youtube")
    payload_after = len(MyPacket.payload)
    MyPacket.len = payload_after
    LabledPacket = IP(dst="137.158.126.22")/TCP()/MyPacket
    #print(LabledPacket.show())
    send(LabledPacket)

    time.sleep(10 - ((time.time()-starttime)%60))

