<template>
  <div id="app">
    <img alt="gpt2 logo" height="200px" width="200px" src="https://avatars.githubusercontent.com/u/14957082?s=400&v=4">
    <router-view/>
    <!--HelloWorld msg="Welcome to Your Vue.js App"/-->
  </div>
</template>

<script>

export default {
  name: 'APP',
};


</script>

<style>

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
