//
//  ViewController.swift
//  malaria_app
//
//  Created by vicente on 3/25/19.
//  Copyright © 2019 vicente. All rights reserved.
//
import AVFoundation
import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var resumeButton: UIButton!
    
    @IBOutlet weak var classLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    private let delayBetweenInferencesMs: Double = 1000
    
    private var result: Inference?
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    
    private lazy var cameraCapture = CameraFeedManager(previewView: previewView)
    
    // Handles all data preprocessing and makes calls to run inference through TfliteWrapper
    private let modelDataHandler: ModelDataHandler? = ModelDataHandler(modelFileName: "malaria_model", labelsFileName: "labels", labelsFileExtension: "txt")
    
    // MARK: View Handling Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Checks if ModelDataHandler initialization was successful to determine if execution should be continued or not.
        guard modelDataHandler != nil else {
            fatalError("Model set up failed")
        }
        
        cameraCapture.delegate = self
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        cameraCapture.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        cameraCapture.stopSession()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    func presentUnableToResumeSessionAlert() {
        let alert = UIAlertController(title: "Unable to Resume Session", message: "There was an error while attempting to resume session.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        self.present(alert, animated: true)
    }
    
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
    
    func didOutput(pixelBuffer: CVPixelBuffer) {
        
        // Run the live camera pixelBuffer through tensorFlow to get the result
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        
        guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
            return
        }

        previousInferenceTimeMs = currentTimeMs
        result = modelDataHandler?.runModel(onFrame: pixelBuffer)
        
        if let classResult = result {
            DispatchQueue.main.async {
                self.classLabel.text = classResult.className
                self.confidenceLabel.text = String(format: "%.2f", classResult.confidence)
            }
        }
        
    }
    
    // MARK: Session Handling Alerts
    func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
        
        // Updates the UI when session is interupted.
        if resumeManually == true {
            self.resumeButton.isHidden = false
        }
    }
    
    func sessionInterruptionEnded() {
        if !self.resumeButton.isHidden {
            self.resumeButton.isHidden = true
        }
    }
    
    func sessionRunTimeErrorOccured() {
        // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
        self.resumeButton.isHidden = false
    }
    
    func presentCameraPermissionsDeniedAlert() {
        let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
            UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
        }
        
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentVideoConfigurationErrorAlert() {
        let alert = UIAlertController(title: "Camera Configuration Failed", message: "There was an error while configuring camera.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        self.present(alert, animated: true)
    }
}
