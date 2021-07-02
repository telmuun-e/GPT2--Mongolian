import axios from 'axios';

const getAPI = axios.create({
    baseURL : 'http://0.0.0.0:8080/',
    timeout : 1000,
})

export {
    getAPI
}