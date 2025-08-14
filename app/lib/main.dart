import 'package:app/pages/splash_screen.dart';
import 'package:flutter/material.dart';

const String CLIENT_ID =
    '1082800166890-fu6fhca38h5b3ftctruo985g669ljlp9.apps.googleusercontent.com';
const String SERVER_CLIENT_ID =
    '910242255946-3okgle3e78inrabcm39807h21cumhvkj.apps.googleusercontent.com';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tacit',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Color(0xFF1C1C1C),
      ),
      home: const SplashScreen(),
    );
  }
}
