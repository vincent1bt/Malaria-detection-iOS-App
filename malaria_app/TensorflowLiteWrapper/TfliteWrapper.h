//
//  TfliteWrapper.h
//  malaria_app
//
//  Created by vicente on 3/25/19.
//  Copyright © 2019 vicente. All rights reserved.
//
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>

/** This class provides Objective C wrapper methods on top of the C++ TensorflowLite library. This
 * wrapper is required because currently there is no interoperability between Swift and C++. This
 * wrapper is exposed to Swift via bridging so that the Tensorflow Lite methods can be called from
 * Swift.
 */
@interface TfliteWrapper : NSObject

/**
 This method initializes the TfliteWrapper with the specified model file.
 */
- (instancetype)initWithModelFileName:(NSString *)fileName;

/**
 This method initializes the interpreter of TensorflowLite library with the specified model file
 that performs the inference.
 */
- (BOOL)setUpModelAndInterpreter;

/**
 This method performs the inference by invoking the interpreter.
 */
- (BOOL)invokeInterpreter;

/**
 This method gets a reference to the input tensor at an index.
 */
- (Float32 *)inputTensorAtIndex:(int)index;

/**
 This method gets the output tensor at a specified index.
 */
- (float *)outputTensorAtIndex:(int)index;

/**
 This method sets the number of threads used by the interpreter to perform inference.
 */
- (void)setNumberOfThreads:(int)threadCount;

@end
