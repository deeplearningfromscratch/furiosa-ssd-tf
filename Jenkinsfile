pipeline {
    agent none
    stages {
        stage('Clone from GitHub') {
            agent {
                label 'master'
            }
            steps {
                withCredentials([string(credentialsId: 'kstreee-github-token', variable: 'TOKEN')]) {
                    checkout(
                        [
                            $class: 'GitSCM',
                            branches: [[name: '*/master']],
                            doGenerateSubmoduleConfigurations: false,
                            extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'furiosa-ssd-tf']],
                            submoduleCfg: [],
                            userRemoteConfigs: [[url: "https://${TOKEN}@github.com/furiosa-ai/furiosa-ssd-tf.git"]]
                        ]
                    )
                }
            }
        }
        stage('Setup venv & build and install target project') {
            agent {
                docker {
                    label 'cpu'
                    image 'docker-registry.furiosa.ai/furiosa-devel/python3'
                }
            }
            steps {
                sh '''#!/bin/bash
                    if [ ! -d "${WORKSPACE}/venv" ];
                    then
                        virtualenv ./venv
                        echo "Set virtualenv"
                    else
                        echo "virtualenv has been set"
                    fi
                '''
                dir("${WORKSPACE}/furiosa-ssd-tf") {
                    sh '''#!/bin/bash
                        source ../venv/bin/activate
                        python setup.py build && python setup.py install
                    '''
		}
            }
        }
        stage('test project') {
            agent {
                docker {
                    label 'cpu'
                    image 'docker-registry.furiosa.ai/furiosa-devel/python3'
                }
            }
            steps {
                dir("${WORKSPACE}/furiosa-ssd-tf") {
                    sh '''#!/bin/bash
                        source ../venv/bin/activate
                        python setup.py test
                    '''
		}
            }
        }
    }
}
