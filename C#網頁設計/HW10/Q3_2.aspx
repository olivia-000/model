<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Q3_2.aspx.cs" Inherits="Q3_2" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        地址：<asp:TextBox ID="txtAddr" runat="server" /><br />
    電話：<asp:TextBox ID="txtTel" runat="server" /><br />
    生日：<asp:TextBox ID="txtBirth" runat="server" Placeholder="yyyy/MM/dd" /><br />
    <asp:Button ID="btnNext" runat="server" Text="下一頁 →" OnClick="btnNext_Click" /><br />

    <asp:RequiredFieldValidator ControlToValidate="txtAddr" runat="server"
        ErrorMessage="*地址必填" />
    <asp:RequiredFieldValidator ControlToValidate="txtTel" runat="server"
        ErrorMessage="*電話必填" />
    <asp:RequiredFieldValidator ControlToValidate="txtBirth" runat="server"
        ErrorMessage="*生日必填" />
    </form>
</body>
</html>
