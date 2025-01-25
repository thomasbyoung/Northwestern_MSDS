import random
from math import exp
import numpy as np

def welcome():
    print()
    print('******************************************************************************')
    print()
    print('Welcome to the Multilayer Perceptron Neural Network')
    print('  trained using the backpropagation method.')
    print('Version 0.2, 01/10/2017, A.J. Maren')
    print('For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu')
    print()
    print('******************************************************************************')
    print()
    return()

def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation   

def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     

def obtainNeuralNetworkSizeSpecs():
    numInputNodes = 2
    numHiddenNodes = 2
    numOutputNodes = 2   
    print(' ')
    print('This network is set up to run the X-OR problem.')
    print('The numbers of nodes in the input, hidden, and output layers have been set to 2 each.') 
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes)
    return (arraySizeList)  

def InitializeWeight():
    randomNum = random.random()
    weight=1-2*randomNum
    return (weight)  

def initializeWeightArray(weightArraySizeList, debugInitializeOff):
    numBottomNodes = weightArraySizeList[0]
    numUpperNodes = weightArraySizeList[1]
    wt00=InitializeWeight()
    wt01=InitializeWeight()
    wt10=InitializeWeight()
    wt11=InitializeWeight()    
    weightArray=np.array([[wt00,wt10],[wt01,wt11]])
    
    if not debugInitializeOff:
        print(' ')
        print('  Inside initializeWeightArray')
        print('    The weights just initialized are: ')
        print('      weight00 = %.4f,' % wt00)
        print('      weight01 = %.4f,' % wt01)
        print('      weight10 = %.4f,' % wt10)
        print('      weight11 = %.4f,' % wt11)
        print(' ')
        print('    The weight Array just established is: ', weightArray)
        print(' ') 
        print('    Within this array: ') 
        print('      weight00 = %.4f    weight10 = %.4f' % (weightArray[0,0], weightArray[0,1]))
        print('      weight01 = %.4f    weight11 = %.4f' % (weightArray[1,0], weightArray[1,1]))    
        print('  Returning to calling procedure')        
    
    return (weightArray)  

def initializeBiasWeightArray(weightArray1DSize):      
    numBiasNodes = weightArray1DSize
    biasWeight0=InitializeWeight()
    biasWeight1=InitializeWeight()
    biasWeightArray=np.array([biasWeight0,biasWeight1])
    return (biasWeightArray)  

def obtainRandomXORTrainingValues():
    trainingDataSetNum = random.randint(1, 4)
    if trainingDataSetNum >1.1:
        if trainingDataSetNum > 2.1:
            if trainingDataSetNum > 3.1:
                trainingDataList = (1,1,0,1,3)
            else: trainingDataList = (1,0,1,0,2)   
        else: trainingDataList = (0,1,1,0,1)     
    else: trainingDataList = (0,0,0,1,0)
    return (trainingDataList)  

def computeSingleNeuronActivation(alpha, wt0, wt1, input0, input1, bias, debugComputeSingleNeuronActivationOff):
    summedNeuronInput = wt0*input0+wt1*input1+bias
    activation = computeTransferFnctn(summedNeuronInput, alpha)

    if not debugComputeSingleNeuronActivationOff:        
        print(' ')
        print('  In computeSingleNeuronActivation with input0, input 1 given as: ', input0, ', ', input1)
        print('    The summed neuron input is %.4f' % summedNeuronInput)   
        print('    The activation (applied transfer function) for that neuron is %.4f' % activation)    
    return activation

def ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray, biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff):
    input0 = inputDataList[0]
    input1 = inputDataList[1]      
    
    wWt00 = wWeightArray[0,0]
    wWt10 = wWeightArray[0,1]
    wWt01 = wWeightArray[1,0]       
    wWt11 = wWeightArray[1,1]
    
    vWt00 = vWeightArray[0,0]
    vWt10 = vWeightArray[0,1]
    vWt01 = vWeightArray[1,0]       
    vWt11 = vWeightArray[1,1]    
    
    biasHidden0 = biasHiddenWeightArray[0]
    biasHidden1 = biasHiddenWeightArray[1]
    biasOutput0 = biasOutputWeightArray[0]
    biasOutput1 = biasOutputWeightArray[1]
    
    if not debugComputeSingleFeedforwardPassOff:
        debugComputeSingleNeuronActivationOff = False
    else: 
        debugComputeSingleNeuronActivationOff = True
        
    if not debugComputeSingleNeuronActivationOff:
        print(' ')
        print('  For hiddenActivation0 from input0, input1 = ', input0, ', ', input1)
    
    hiddenActivation0 = computeSingleNeuronActivation(alpha, wWt00, wWt10, input0, input1, biasHidden0,
    debugComputeSingleNeuronActivationOff)
    
    if not debugComputeSingleNeuronActivationOff:
        print(' ')
        print('  For hiddenActivation1 from input0, input1 = ', input0, ', ', input1)    
    
    hiddenActivation1 = computeSingleNeuronActivation(alpha, wWt01, wWt11, input0, input1, biasHidden1,
    debugComputeSingleNeuronActivationOff)
    
    if not debugComputeSingleFeedforwardPassOff: 
        print(' ')
        print('  In computeSingleFeedforwardPass: ')
        print('  Input node values: ', input0, ', ', input1)
        print('  The activations for the hidden nodes are:')
        print('    Hidden0 = %.4f' % hiddenActivation0, 'Hidden1 = %.4f' % hiddenActivation1)

    outputActivation0 = computeSingleNeuronActivation(alpha, vWt00, vWt10, hiddenActivation0, 
    hiddenActivation1, biasOutput0, debugComputeSingleNeuronActivationOff)
    outputActivation1 = computeSingleNeuronActivation(alpha, vWt01, vWt11, hiddenActivation0, 
    hiddenActivation1, biasOutput1, debugComputeSingleNeuronActivationOff)
    
    if not debugComputeSingleFeedforwardPassOff: 
        print(' ')
        print('  Computing the output neuron activations') 
        print(' ')        
        print('  Back in ComputeSingleFeedforwardPass (for hidden-to-output computations)')
        print('  The activations for the output nodes are:')
        print('    Output0 = %.4f' % outputActivation0, 'Output1 = %.4f' % outputActivation1)
               
    actualAllNodesOutputList = (hiddenActivation0, hiddenActivation1, outputActivation0, outputActivation1)
                                                                                                
    return actualAllNodesOutputList

def computeSSE_Values(alpha, SSE_InitialArray, wWeightArray, vWeightArray, biasHiddenWeightArray, biasOutputWeightArray, debugSSE_InitialComputationOff):
    if not debugSSE_InitialComputationOff:
        debugComputeSingleFeedforwardPassOff = False
    else:
        debugComputeSingleFeedforwardPassOff = True
              
    inputDataList = (0, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray, 
    biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList[2]
    actualOutput1 = actualAllNodesOutputList[3]
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[0] = error0**2 + error1**2

    if not debugSSE_InitialComputationOff:
        input0 = inputDataList[0]
        input1 = inputDataList[1]
        print(' ')
        print('  In computeSSE_Values')
        print(' ')
        print('  Actual Node Outputs for (0,0) training set:')
        print('     input0 = ', input0, '   input1 = ', input1)
        print('     actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('     error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('  Initial SSE for (0,0) = %.4f' % SSE_InitialArray[0])

    inputDataList = (0, 1)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray,
    biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList[2]
    actualOutput1 = actualAllNodesOutputList[3]
    error0 = 1.0 - actualOutput0
    error1 = 0.0 - actualOutput1
    SSE_InitialArray[1] = error0**2 + error1**2

    if not debugSSE_InitialComputationOff:
        input0 = inputDataList[0]
        input1 = inputDataList[1]
        print(' ')
        print('  Actual Node Outputs for (0,1) training set:')
        print('     input0 = ', input0, '   input1 = ', input1)
        print('     actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('     error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('  Initial SSE for (0,1) = %.4f' % SSE_InitialArray[1])
                                                            
    inputDataList = (1, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray,
    biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList[2]
    actualOutput1 = actualAllNodesOutputList[3]
    error0 = 1.0 - actualOutput0
    error1 = 0.0 - actualOutput1
    SSE_InitialArray[2] = error0**2 + error1**2
    
    if not debugSSE_InitialComputationOff:
        input0 = inputDataList[0]
        input1 = inputDataList[1]
        print(' ')
        print('  Actual Node Outputs for (1,0) training set:')
        print('     input0 = ', input0, '   input1 = ', input1)
        print('     actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('     error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('  Initial SSE for (1,0) = %.4f' % SSE_InitialArray[2])
            
    inputDataList = (1, 1)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray,
    biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList[2]
    actualOutput1 = actualAllNodesOutputList[3]
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1 
    SSE_InitialArray[3] = error0**2 + error1**2

    if not debugSSE_InitialComputationOff:
        input0 = inputDataList[0]
        input1 = inputDataList[1]
        print(' ')
        print('  Actual Node Outputs for (1,1) training set:')
        print('     input0 = ', input0, '   input1 = ', input1)
        print('     actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('     error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('  Initial SSE for (1,1) = %.4f' % SSE_InitialArray[3])
    
    SSE_InitialTotal = SSE_InitialArray[0] + SSE_InitialArray[1] + SSE_InitialArray[2] + SSE_InitialArray[3]

    if not debugSSE_InitialComputationOff:
        print(' ')
        print('  The initial total of the SSEs is %.4f' %SSE_InitialTotal)

    SSE_InitialArray[4] = SSE_InitialTotal
    return SSE_InitialArray

def PrintAndTraceBackpropagateOutputToHidden(alpha, eta, errorList, actualAllNodesOutputList, transFuncDerivList, deltaVWtArray, vWeightArray, newVWeightArray):
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    transFuncDeriv0 = transFuncDerivList[0]
    transFuncDeriv1 = transFuncDerivList[1]

    deltaVWt00 = deltaVWtArray[0,0]
    deltaVWt01 = deltaVWtArray[1,0]
    deltaVWt10 = deltaVWtArray[0,1]
    deltaVWt11 = deltaVWtArray[1,1]    
    
    error0 = errorList[0]
    error1 = errorList[1]                 
        
    print(' ')
    print('In Print and Trace for Backpropagation: Hidden to Output Weights')
    print('  Assuming alpha = 1')
    print(' ')
    print('  The hidden node activations are:')
    print('    Hidden node 0: ', '  %.4f' % hiddenNode0, '  Hidden node 1: ', '  %.4f' % hiddenNode1)   
    print(' ')
    print('  The output node activations are:')
    print('    Output node 0: ', '  %.3f' % outputNode0, '   Output node 1: ', '  %.3f' % outputNode1)       
    print(' ') 
    print('  The transfer function derivatives are: ')
    print('    Deriv-F(0): ', '     %.3f' % transFuncDeriv0, '   Deriv-F(1): ', '     %.3f' % transFuncDeriv1)

    print(' ') 
    print('The computed values for the deltas are: ')
    print('                eta  *  error  *   trFncDeriv *   hidden')
    print('  deltaVWt00 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode0)
    print('  deltaVWt01 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode0)                       
    print('  deltaVWt10 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode1)
    print('  deltaVWt11 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode1)
    print(' ')
    print('Values for the hidden-to-output connection weights:')
    print('           Old:     New:      eta*Delta:')
    print('[0,0]:   %.4f' % vWeightArray[0,0], '  %.4f' % newVWeightArray[0,0], '  %.4f' % deltaVWtArray[0,0])
    print('[0,1]:   %.4f' % vWeightArray[1,0], '  %.4f' % newVWeightArray[1,0], '  %.4f' % deltaVWtArray[1,0])
    print('[1,0]:   %.4f' % vWeightArray[0,1], '  %.4f' % newVWeightArray[0,1], '  %.4f' % deltaVWtArray[0,1])
    print('[1,1]:   %.4f' % vWeightArray[1,1], '  %.4f' % newVWeightArray[1,1], '  %.4f' % deltaVWtArray[1,1])

def BackpropagateOutputToHidden(alpha, eta, errorList, actualAllNodesOutputList, vWeightArray):
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1]  
    
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]  
        
    transFuncDeriv0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 

    partialSSE_w_Vwt00 = -error0*transFuncDeriv0*hiddenNode0                                                             
    partialSSE_w_Vwt01 = -error1*transFuncDeriv1*hiddenNode0
    partialSSE_w_Vwt10 = -error0*transFuncDeriv0*hiddenNode1
    partialSSE_w_Vwt11 = -error1*transFuncDeriv1*hiddenNode1                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                
    deltaVWt00 = -eta*partialSSE_w_Vwt00
    deltaVWt01 = -eta*partialSSE_w_Vwt01        
    deltaVWt10 = -eta*partialSSE_w_Vwt10
    deltaVWt11 = -eta*partialSSE_w_Vwt11 
    deltaVWtArray = np.array([[deltaVWt00, deltaVWt10],[deltaVWt01, deltaVWt11]])

    vWt00 = vWt00+deltaVWt00
    vWt01 = vWt01+deltaVWt01
    vWt10 = vWt10+deltaVWt10
    vWt11 = vWt11+deltaVWt11 
    
    newVWeightArray = np.array([[vWt00, vWt10], [vWt01, vWt11]])

    PrintAndTraceBackpropagateOutputToHidden(alpha, eta, errorList, actualAllNodesOutputList, transFuncDerivList, deltaVWtArray, vWeightArray, newVWeightArray)    
                                                                                                                                            
    return newVWeightArray

def BackpropagateBiasOutputWeights(alpha, eta, errorList, actualAllNodesOutputList, biasOutputWeightArray):
    error0 = errorList[0]
    error1 = errorList[1]

    biasOutputWt0 = biasOutputWeightArray[0]
    biasOutputWt1 = biasOutputWeightArray[1]
    
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]  
              
    transFuncDeriv0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 
    
    partialSSE_w_BiasOutput0 = -error0*transFuncDeriv0
    partialSSE_w_BiasOutput1 = -error1*transFuncDeriv1    
                                                                                                                                                                                                                                                                
    deltaBiasOutput0 = -eta*partialSSE_w_BiasOutput0
    deltaBiasOutput1 = -eta*partialSSE_w_BiasOutput1

    biasOutputWt0 = biasOutputWt0+deltaBiasOutput0
    biasOutputWt1 = biasOutputWt1+deltaBiasOutput1 

    newBiasOutputWeightArray = np.array([biasOutputWt0, biasOutputWt1])
                                                                                                                                            
    return newBiasOutputWeightArray

def BackpropagateHiddenToInput(alpha, eta, errorList, actualAllNodesOutputList, inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1]  
    
    wWt00 = wWeightArray[0,0]
    wWt01 = wWeightArray[1,0]
    wWt10 = wWeightArray[0,1]       
    wWt11 = wWeightArray[1,1] 
    
    inputNode0 = inputDataList[0] 
    inputNode1 = inputDataList[1]         
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
        
    transFuncDerivHidden0 = computeTransferFnctnDeriv(hiddenNode0, alpha) 
    transFuncDerivHidden1 = computeTransferFnctnDeriv(hiddenNode1, alpha)
    transFuncDerivHiddenList = (transFuncDerivHidden0, transFuncDerivHidden1) 
    
    transFuncDerivOutput0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDerivOutput1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivOutputList = (transFuncDerivOutput0, transFuncDerivOutput1) 
    
    errorTimesTransDOutput0 = error0*transFuncDerivOutput0
    errorTimesTransDOutput1 = error1*transFuncDerivOutput1
               
    partialSSE_w_Wwt00 = -transFuncDerivHidden0*inputNode0*(vWt00*errorTimesTransDOutput0 + vWt01*errorTimesTransDOutput1)                                                             
    partialSSE_w_Wwt01 = -transFuncDerivHidden1*inputNode0*(vWt10*errorTimesTransDOutput0 + vWt11*errorTimesTransDOutput1)
    partialSSE_w_Wwt10 = -transFuncDerivHidden0*inputNode1*(vWt00*errorTimesTransDOutput0 + vWt01*errorTimesTransDOutput1)
    partialSSE_w_Wwt11 = -transFuncDerivHidden1*inputNode1*(vWt10*errorTimesTransDOutput0 + vWt11*errorTimesTransDOutput1)                                                                                                    
                                                                                                                                                                                                                                                
    deltaWWt00 = -eta*partialSSE_w_Wwt00
    deltaWWt01 = -eta*partialSSE_w_Wwt01        
    deltaWWt10 = -eta*partialSSE_w_Wwt10
    deltaWWt11 = -eta*partialSSE_w_Wwt11 
    deltaWWtArray = np.array([[deltaWWt00, deltaWWt10],[deltaWWt01, deltaWWt11]])

    wWt00 = wWt00+deltaWWt00
    wWt01 = wWt01+deltaWWt01
    wWt10 = wWt10+deltaWWt10
    wWt11 = wWt11+deltaWWt11 
    
    newWWeightArray = np.array([[wWt00, wWt10], [wWt01, wWt11]])
                                                                    
    return newWWeightArray

def BackpropagateBiasHiddenWeights(alpha, eta, errorList, actualAllNodesOutputList, vWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1]      

    biasHiddenWt0 = biasHiddenWeightArray[0]
    biasHiddenWt1 = biasHiddenWeightArray[1]    
    biasOutputWt0 = biasOutputWeightArray[0]
    biasOutputWt1 = biasOutputWeightArray[1]
    
    hiddenNode0= actualAllNodesOutputList[0]
    hiddenNode1= actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]  
    
    transFuncDerivOutput0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDerivOutput1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivHidden0 = computeTransferFnctnDeriv(hiddenNode0, alpha) 
    transFuncDerivHidden1 = computeTransferFnctnDeriv(hiddenNode1, alpha)    

    transFuncDerivOutputList = (transFuncDerivOutput0, transFuncDerivOutput1) 

    errorTimesTransDOutput0 = error0*transFuncDerivOutput0
    errorTimesTransDOutput1 = error1*transFuncDerivOutput1
    
    partialSSE_w_BiasHidden0 = -transFuncDerivHidden0*(errorTimesTransDOutput0*vWt00 + errorTimesTransDOutput1*vWt01)
    partialSSE_w_BiasHidden1 = -transFuncDerivHidden1*(errorTimesTransDOutput0*vWt10 + errorTimesTransDOutput1*vWt11)  
                                                                                                                                                                                                                                                                
    deltaBiasHidden0 = -eta*partialSSE_w_BiasHidden0
    deltaBiasHidden1 = -eta*partialSSE_w_BiasHidden1

    biasHiddenWt0 = biasHiddenWt0+deltaBiasHidden0
    biasHiddenWt1 = biasHiddenWt1+deltaBiasHidden1 
        
    newBiasHiddenWeightArray = np.array([biasHiddenWt0, biasHiddenWt1])
                                                                                                                                            
    return newBiasHiddenWeightArray


def main():
    welcome()
    
    alpha = 1.0             
    maxNumIterations = 5000    
    eta = 0.5               

    arraySizeList = list() 
       
    arraySizeList = obtainNeuralNetworkSizeSpecs()
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    trainingDataList = (0,0,0,0,0)
           
    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength      

    debugCallInitializeOff = True
    debugInitializeOff = True
    wWeightArray = initializeWeightArray(wWeightArraySizeList, debugInitializeOff)
    vWeightArray = initializeWeightArray(vWeightArraySizeList, debugInitializeOff)

    biasHiddenWeightArray = initializeBiasWeightArray(biasHiddenWeightArraySize)
    biasOutputWeightArray = initializeBiasWeightArray(biasOutputWeightArraySize) 

    initialWWeightArray = wWeightArray[:]
    initialVWeightArray = vWeightArray[:]
    initialBiasHiddenWeightArray = biasHiddenWeightArray[:]   
    initialBiasOutputWeightArray = biasOutputWeightArray[:] 
    
    print()
    print('The initial weights for this neural network are:')
    print('       Input-to-Hidden                            Hidden-to-Output')
    print('  w(0,0) = %.4f   w(1,0) = %.4f         v(0,0) = %.4f   v(1,0) = %.4f' % (initialWWeightArray[0,0], 
    initialWWeightArray[0,1], initialVWeightArray[0,0], initialVWeightArray[0,1]))
    print('  w(0,1) = %.4f   w(1,1) = %.4f         v(0,1) = %.4f   v(1,1) = %.4f' % (initialWWeightArray[1,0], 
    initialWWeightArray[1,1], initialVWeightArray[1,0], initialVWeightArray[1,1])) 
    print(' ')
    print('       Bias at Hidden Layer                          Bias at Output Layer')
    print('       b(hidden,0) = %.4f                           b(output,0) = %.4f' % (biasHiddenWeightArray[0],
    biasOutputWeightArray[0]))                  
    print('       b(hidden,1) = %.4f                           b(output,1) = %.4f' % (biasHiddenWeightArray[1],
    biasOutputWeightArray[1]))  
  
    epsilon = 0.2
    iteration = 0
    SSE_InitialTotal = 0.0
                        
    SSE_InitialArray = [0,0,0,0,0]
    debugSSE_InitialComputationOff = True

    SSE_InitialArray = computeSSE_Values(alpha, SSE_InitialArray, wWeightArray, vWeightArray, 
    biasHiddenWeightArray, biasOutputWeightArray, debugSSE_InitialComputationOff)    

    SSE_Array = SSE_InitialArray[:] 
    SSE_InitialTotal = SSE_Array[4] 
    
    debugSSE_InitialComputationReportOff = True    
    
    while iteration < maxNumIterations: 
        trainingDataList = obtainRandomXORTrainingValues() 
        input0 = trainingDataList[0]
        input1 = trainingDataList[1] 
        desiredOutput0 = trainingDataList[2]
        desiredOutput1 = trainingDataList[3]
        setNumber = trainingDataList[4]       
        print(' ')
        print('Randomly selecting XOR inputs for XOR, identifying desired outputs for this training pass:')
        print('          Input0 = ', input0, '            Input1 = ', input1)   
        print(' Desired Output0 = ', desiredOutput0, '   Desired Output1 = ', desiredOutput1)    
        print(' ')
         
        errorList = (0,0)
        actualAllNodesOutputList = (0,0,0,0)     
        inputDataList = (input0, input1)         
        debugComputeSingleFeedforwardPassOff = True
        
        actualAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray, 
        biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)

        actualHiddenOutput0 = actualAllNodesOutputList[0] 
        actualHiddenOutput1 = actualAllNodesOutputList[1] 
        actualOutput0 = actualAllNodesOutputList[2]
        actualOutput1 = actualAllNodesOutputList[3] 
    
        error0 = desiredOutput0 - actualOutput0
        error1 = desiredOutput1 - actualOutput1
        errorList = (error0, error1)
        SSEInitial = error0**2 + error1**2
        
        debugMainComputeForwardPassOutputsOff = True
   
        newVWeightArray = BackpropagateOutputToHidden(alpha, eta, errorList, actualAllNodesOutputList, vWeightArray)

        newBiasOutputWeightArray = BackpropagateBiasOutputWeights(alpha, eta, errorList, actualAllNodesOutputList, 
        biasOutputWeightArray)
        newBiasOutputWeight0 = newBiasOutputWeightArray[0]
        newBiasOutputWeight1 = newBiasOutputWeightArray[1]
        
        newWWeightArray = BackpropagateHiddenToInput(alpha, eta, errorList, actualAllNodesOutputList, inputDataList, 
        vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)

        newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights(alpha, eta, errorList, actualAllNodesOutputList, 
        vWeightArray, biasHiddenWeightArray, biasOutputWeightArray)
        newBiasHiddenWeight0 = newBiasHiddenWeightArray[0]
        newBiasHiddenWeight1 = newBiasHiddenWeightArray[1]        
        
        newBiasWeightArray = [[newBiasOutputWeight0, newBiasOutputWeight1], [newBiasHiddenWeight0, newBiasHiddenWeight1]] 

        debugWeightArrayOff = False
        if not debugWeightArrayOff:
            print(' ')
            print('    The weights before backpropagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' % (wWeightArray[0,0], 
            wWeightArray[0,1], vWeightArray[0,0], vWeightArray[0,1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' % (wWeightArray[1,0], 
            wWeightArray[1,1], vWeightArray[1,0], vWeightArray[1,1]))             
            print(' ')
            print('    The weights after backpropagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' % (newWWeightArray[0,0], 
            newWWeightArray[0,1], newVWeightArray[0,0], newVWeightArray[0,1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' % (newWWeightArray[1,0], 
            newWWeightArray[1,1], newVWeightArray[1,0], newVWeightArray[1,1]))
            
        vWeightArray = newVWeightArray[:]
        wWeightArray = newWWeightArray[:]
    
        newAllNodesOutputList = ComputeSingleFeedforwardPass(alpha, inputDataList, wWeightArray, vWeightArray,
        biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)         
        newOutput0 = newAllNodesOutputList[2]
        newOutput1 = newAllNodesOutputList[3] 

        newError0 = desiredOutput0 - newOutput0
        newError1 = desiredOutput1 - newOutput1
        newErrorList = (newError0, newError1)

        SSE0 = newError0**2
        SSE1 = newError1**2
        newSSE = SSE0 + SSE1

        SSE_Array[setNumber] = newSSE

        previousSSE_Total = SSE_Array[4]
        print(' ') 
        print('  The previous SSE Total was %.4f' % previousSSE_Total)

        newSSE_Total = SSE_Array[0] + SSE_Array[1] + SSE_Array[2] + SSE_Array[3]
        print('  The new SSE Total was %.4f' % newSSE_Total)
        print('    For node 0: Desired Output = ',desiredOutput0,  ' New Output = %.4f' % newOutput0) 
        print('    For node 1: Desired Output = ',desiredOutput1,  ' New Output = %.4f' % newOutput1)  
        print('    Error(0) = %.4f,   Error(1) = %.4f' %(newError0, newError1))
        print('    SSE0(0) =   %.4f,   SSE(1) =   %.4f' % (SSE0, SSE1))                

        SSE_Array[4] = newSSE_Total
        deltaSSE = previousSSE_Total - newSSE_Total
        print('  Delta in the SSEs is %.4f' % deltaSSE) 
        if deltaSSE > 0:
            print('SSE improvement')
        else: print('NO improvement')     
                         
        errorList = newErrorList[:]
        
        print(' ')
        print('Iteration number ', iteration)
        iteration = iteration + 1

        if newSSE_Total < epsilon:
            break
            
    print('Out of while loop')     

    debugEndingSSEComparisonOff = False
    if not debugEndingSSEComparisonOff:
        SSE_Array[4] = newSSE_Total
        deltaSSE = previousSSE_Total - newSSE_Total
        print('  Initial Total SSE = %.4f'  % SSE_InitialTotal)
        print('  Final Total SSE = %.4f'  % newSSE_Total)
        finalDeltaSSE = SSE_InitialTotal - newSSE_Total
        print('  Delta in the SSEs is %.4f' % finalDeltaSSE) 
        if finalDeltaSSE > 0:
            print('SSE total improvement')
        else: print('NO improvement in total SSE')

if __name__ == "__main__": main()
   