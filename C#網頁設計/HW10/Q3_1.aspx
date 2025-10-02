<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Q3_1.aspx.cs" Inherits="Q3_1" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        使用者名稱：<asp:TextBox ID="txtUser" runat="server" /><br />
    密碼：<asp:TextBox ID="txtPwd" runat="server" TextMode="Password" /><br />
    <asp:Button ID="btnNext" runat="server" Text="下一頁 →" OnClick="btnNext_Click" />
    <asp:RequiredFieldValidator ControlToValidate="txtUser" runat="server"
        ErrorMessage="*帳號必填" /><br />
    <asp:RequiredFieldValidator ControlToValidate="txtPwd" runat="server"
        ErrorMessage="*密碼必填" />
    </form>
</body>
</html>
