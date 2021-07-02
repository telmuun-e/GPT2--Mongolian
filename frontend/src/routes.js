import Vue from 'vue';
import VueRouter from 'vue-router';
import Prediction from './components/Prediction.vue';

Vue.use(VueRouter)

export default new VueRouter({
    //base : "http://127.0.0.1:5000",
    mode : "history",
    routes : [{
        path : '/',
        name : 'Prediction',
        component : Prediction,
    }]
})